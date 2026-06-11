from __future__ import annotations

import json
import math
import re
import subprocess
from fractions import Fraction
from pathlib import Path


def run_process(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=True)


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


def extract_frame(video_path: Path, seconds: float, output_path: Path) -> None:
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
        ]
    )


def extract_frames(video_path: Path, output_dir: Path) -> list[Path]:
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
        ]
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


def encode_video(
    frames_dir: Path,
    source_video: Path,
    output_path: Path,
    fps: float,
    crf: int,
    encoder: str = "auto",
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = encoder
    if selected == "auto":
        selected = "h264_nvenc" if h264_nvenc_available() else "libx264"

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

    if selected == "h264_nvenc":
        video_args = [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p5",
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
    else:
        selected = "libx264"
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

    try:
        run_process(base_args + video_args + tail_args)
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
            run_process(base_args + video_args + tail_args)
        else:
            raise
    return selected

