from __future__ import annotations

import json
import math
import re
import subprocess
from fractions import Fraction
from pathlib import Path

from .resource_control import process_creationflags


def run_process(
    args: list[str],
    cwd: Path | None = None,
    *,
    low_priority: bool = False,
) -> subprocess.CompletedProcess:
    kwargs = {
        "cwd": cwd,
        "text": True,
        "capture_output": True,
        "check": True,
    }
    creationflags = process_creationflags(low_priority)
    if creationflags:
        kwargs["creationflags"] = creationflags
    return subprocess.run(args, **kwargs)


def safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return name[:120] or "video"


def parse_rate(value: str | None) -> float:
    if not value or value == "0/0":
        return 30.0
    try:
        return float(Fraction(value))
    except Exception:
        return 30.0


def probe_video(path: Path) -> dict:
    proc = run_process(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    data = json.loads(proc.stdout)
    video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if not video_stream:
        raise ValueError("No video stream found.")
    audio_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    duration = float(video_stream.get("duration") or data.get("format", {}).get("duration") or 0)
    fps = parse_rate(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))
    frame_count = video_stream.get("nb_frames")
    if not frame_count and duration > 0:
        frame_count = str(math.ceil(duration * fps))
    return {
        "duration": duration,
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "fps": fps,
        "frameCount": int(frame_count or 0),
        "codec": video_stream.get("codec_name"),
        "pixelFormat": video_stream.get("pix_fmt"),
        "hasAudio": audio_stream is not None,
        "audioCodec": audio_stream.get("codec_name") if audio_stream else None,
    }


def extract_frame(video_path: Path, seconds: float, output_path: Path, *, low_priority: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_process(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(0.0, seconds):.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_path),
        ],
        low_priority=low_priority,
    )


def extract_frames(video_path: Path, output_dir: Path, *, low_priority: bool = False) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob("frame_*.png"):
        existing.unlink()
    run_process(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-q:v",
            "2",
            str(output_dir / "frame_%06d.png"),
        ],
        low_priority=low_priority,
    )
    frames = sorted(output_dir.glob("frame_*.png"))
    if not frames:
        raise RuntimeError("ffmpeg did not extract any frames.")
    return frames


def h264_nvenc_available() -> bool:
    try:
        proc = run_process(["ffmpeg", "-hide_banner", "-encoders"])
    except Exception:
        return False
    return "h264_nvenc" in proc.stdout


def h264_video_args(encoder: str, crf: int, *, nvenc_preset: str = "p5", x264_preset: str = "slow") -> tuple[str, list[str]]:
    selected = encoder
    if selected == "auto":
        selected = "h264_nvenc" if h264_nvenc_available() else "libx264"

    if selected == "h264_nvenc":
        return selected, [
            "-c:v",
            "h264_nvenc",
            "-preset",
            nvenc_preset,
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            str(crf),
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
        ]

    return "libx264", [
        "-c:v",
        "libx264",
        "-preset",
        x264_preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
    ]


def encode_video(
    frames_dir: Path,
    source_video: Path,
    output_path: Path,
    fps: float,
    crf: int,
    encoder: str = "auto",
    frame_count: int | None = None,
    low_priority: bool = False,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_args = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        f"{fps:.6f}",
        "-i",
        str(frames_dir / "frame_%06d.png"),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
    ]
    tail_args = ["-c:a", "copy", "-shortest", "-movflags", "+faststart", str(output_path)]
    if frame_count is not None:
        tail_args = ["-frames:v", str(frame_count)] + tail_args

    selected, video_args = h264_video_args(encoder, crf)

    try:
        run_process(base_args + video_args + tail_args, low_priority=low_priority)
    except subprocess.CalledProcessError:
        if selected == "h264_nvenc":
            selected = "libx264"
            output_path.unlink(missing_ok=True)
            video_args = [
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
            ]
            run_process(base_args + video_args + tail_args, low_priority=low_priority)
        else:
            raise
    return selected


def create_video_chunk(
    video_path: Path,
    output_path: Path,
    start_frame: int,
    frame_count: int,
    fps: float,
    longest_side_cap: int | None = None,
    min_frame_count: int | None = None,
    low_priority: bool = False,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_seconds = max(0.0, start_frame / max(1.0, fps))
    args = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_seconds:.6f}",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
    ]
    filters = []
    if longest_side_cap is not None and longest_side_cap > 0:
        safe_longest = max(256, int(longest_side_cap))
        filters.append(
            f"scale=w=min({safe_longest}\\,iw):h=min({safe_longest}\\,ih):"
            "force_original_aspect_ratio=decrease:force_divisible_by=2"
        )
    output_frame_count = max(1, frame_count)
    if min_frame_count is not None and min_frame_count > output_frame_count:
        pad_duration = (min_frame_count - output_frame_count + 1) / max(1.0, fps)
        filters.append(f"tpad=stop_mode=clone:stop_duration={pad_duration:.6f}")
        output_frame_count = min_frame_count
    if filters:
        args.extend(["-vf", ",".join(filters)])
    args.extend(
        [
            "-an",
            "-frames:v",
            str(output_frame_count),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "10",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
    )
    run_process(args, low_priority=low_priority)
    return probe_video(output_path)


def trim_video_frame_count(input_path: Path, output_path: Path, frame_count: int, *, low_priority: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-an",
        "-frames:v",
        str(max(1, frame_count)),
        "-c",
        "copy",
        str(output_path),
    ]
    try:
        run_process(args, low_priority=low_priority)
    except subprocess.CalledProcessError:
        run_process(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(input_path),
                "-map",
                "0:v:0",
                "-an",
                "-frames:v",
                str(max(1, frame_count)),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "10",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ],
            low_priority=low_priority,
        )


def concat_videos_copy(video_paths: list[Path], output_path: Path, *, low_priority: bool = False) -> None:
    if not video_paths:
        raise RuntimeError("No video chunks to concatenate.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_path = output_path.parent / f"{output_path.stem}_concat.txt"
    lines = []
    for path in video_paths:
        relative = path.resolve().as_posix()
        escaped = relative.replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_process(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        low_priority=low_priority,
    )


def remux_video_with_source_audio(
    processed_video: Path,
    source_video: Path,
    output_path: Path,
    *,
    low_priority: bool = False,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(processed_video),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        run_process(args, low_priority=low_priority)
        return "video copy, audio copy"
    except subprocess.CalledProcessError:
        output_path.unlink(missing_ok=True)
        fallback = args[:-6] + [
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        run_process(fallback, low_priority=low_priority)
        return "video copy, audio aac"
