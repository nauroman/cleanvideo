from __future__ import annotations

import os
import random
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .resource_control import process_creationflags
from .video_ops import probe_video


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DoveSettings:
    scale_by: str = "factor"
    upscale: int = 1
    target_longest_side: int | None = None
    seed: int = 231
    dtype: str = "bfloat16"
    chunk_len: int = 0
    overlap_t: int = 8
    tile_height: int = 0
    tile_width: int = 0
    overlap_h: int = 32
    overlap_w: int = 32
    cpu_offload: bool = False


class DoveEngine:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else ROOT

    @property
    def python_path(self) -> Path:
        return self.root / ".venv-dove" / "Scripts" / "python.exe"

    @property
    def repo_path(self) -> Path:
        return self.root / "external" / "DOVE"

    @property
    def cli_path(self) -> Path:
        return self.repo_path / "inference_script.py"

    @property
    def model_path(self) -> Path:
        return self.repo_path / "pretrained_models" / "DOVE"

    @property
    def prompt_embedding_path(self) -> Path:
        return (
            self.repo_path
            / "pretrained_models"
            / "prompt_embeddings"
            / "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.safetensors"
        )

    def _required_model_paths(self) -> list[Path]:
        return [
            self.model_path / "model_index.json",
            self.model_path / "scheduler",
            self.model_path / "text_encoder",
            self.model_path / "tokenizer",
            self.model_path / "transformer",
            self.model_path / "vae",
        ]

    def status(self) -> dict:
        missing = []
        for path in [self.python_path, self.cli_path, self.model_path]:
            if not path.exists():
                missing.append(str(path))
        if self.model_path.exists():
            for path in self._required_model_paths():
                if not path.exists():
                    missing.append(str(path))
        available = not missing
        return {
            "available": available,
            "loaded": False,
            "repoPath": str(self.repo_path),
            "pythonPath": str(self.python_path),
            "scriptPath": str(self.cli_path),
            "modelPath": str(self.model_path),
            "promptEmbeddingPath": str(self.prompt_embedding_path),
            "promptEmbeddingPresent": self.prompt_embedding_path.exists(),
            "missing": missing,
            "adapterSupport": "not_available",
            "adapterNote": (
                "DOVE is a video-native inference path. This app does not expose a local "
                "video-to-adapter workflow for DOVE."
            ),
            "blockedReason": None if available else "DOVE runtime, checkout, or pretrained model is missing.",
        }

    def _ensure_available(self) -> None:
        status = self.status()
        if not status["available"]:
            missing = ", ".join(status["missing"]) or "unknown DOVE dependency"
            raise RuntimeError(f"DOVE is not ready. Missing: {missing}")

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["TOKENIZERS_PARALLELISM"] = "false"
        return env

    def output_for_input(self, output_dir: Path, input_path: Path) -> Path:
        name = input_path.name
        if input_path.suffix.lower() == ".mkv":
            name = f"{input_path.stem}.mp4"
        return output_dir / name

    def build_command(
        self,
        input_path: Path,
        output_path: Path,
        settings: DoveSettings,
        *,
        work_dir: Path,
        fps: float | None = None,
        frame_count: int | None = None,
    ) -> list[str]:
        input_dir = work_dir / "input"
        output_dir = work_dir / "output"
        seed = settings.seed if settings.seed != -1 else random.randint(0, 2**32 - 1)
        args = [
            str(self.python_path),
            str(self.cli_path),
            "--input_dir",
            str(input_dir),
            "--model_path",
            str(self.model_path),
            "--output_path",
            str(output_dir),
            "--dtype",
            settings.dtype,
            "--seed",
            str(seed),
            "--upscale",
            str(max(1, min(4, int(settings.upscale or 1)))),
            "--is_vae_st",
            "--save_format",
            "yuv420p",
        ]
        if fps is not None and fps > 0:
            args.extend(["--fps", str(max(1, int(round(fps))))])
        if settings.cpu_offload:
            args.append("--is_cpu_offload")
        if settings.chunk_len > 0 and (frame_count is None or frame_count > settings.chunk_len):
            args.extend(["--chunk_len", str(settings.chunk_len), "--overlap_t", str(settings.overlap_t)])
        if settings.tile_height > 0 and settings.tile_width > 0:
            args.extend(
                [
                    "--tile_size_hw",
                    str(settings.tile_height),
                    str(settings.tile_width),
                    "--overlap_hw",
                    str(settings.overlap_h),
                    str(settings.overlap_w),
                ]
            )
        return args

    def _prepare_isolated_input(self, input_path: Path, work_dir: Path) -> Path:
        input_dir = work_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        isolated = input_dir / "input.mp4"
        isolated.unlink(missing_ok=True)
        try:
            os.link(input_path, isolated)
        except OSError:
            shutil.copy2(input_path, isolated)
        return isolated

    def enhance_video(
        self,
        input_path: Path,
        output_path: Path,
        settings: DoveSettings,
        *,
        fps: float | None = None,
        on_line: Callable[[str], None] | None = None,
        on_process: Callable[[subprocess.Popen], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        low_priority: bool = False,
    ) -> dict:
        self._ensure_available()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        work_dir = output_path.parent / f"{output_path.stem}_dove_{uuid.uuid4().hex[:8]}"
        output_dir = work_dir / "output"
        try:
            isolated_input = self._prepare_isolated_input(input_path, work_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            frame_count = None
            try:
                frame_count = int(probe_video(input_path).get("frameCount") or 0) or None
            except Exception:
                frame_count = None
            args = self.build_command(
                isolated_input,
                output_path,
                settings,
                work_dir=work_dir,
                fps=fps,
                frame_count=frame_count,
            )
            process = subprocess.Popen(
                args,
                cwd=str(self.repo_path),
                env=self._env(),
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=process_creationflags(low_priority),
            )
            if on_process:
                on_process(process)
            assert process.stdout is not None
            output_tail: list[str] = []
            for line in process.stdout:
                stripped = line.rstrip()
                if stripped:
                    output_tail.append(stripped)
                    output_tail = output_tail[-20:]
                if on_line:
                    on_line(stripped)
                if should_cancel and should_cancel():
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise RuntimeError("DOVE export cancelled")
            return_code = process.wait()
            if should_cancel and should_cancel():
                raise RuntimeError("DOVE export cancelled")
            if return_code != 0:
                detail = "\n".join(output_tail[-10:])
                suffix = f"\nLast output:\n{detail}" if detail else ""
                raise RuntimeError(f"DOVE failed with exit code {return_code}{suffix}")

            generated = self.output_for_input(output_dir, isolated_input)
            if not generated.exists() or generated.stat().st_size == 0:
                raise RuntimeError(f"DOVE did not create output: {generated}")
            output_path.unlink(missing_ok=True)
            shutil.move(str(generated), str(output_path))
            return {"seed": settings.seed, "width": None, "height": None}
        finally:
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)


engine = DoveEngine()
