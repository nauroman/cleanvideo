from __future__ import annotations

import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

from .video_ops import probe_video


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SeedVr2Settings:
    scale_by: str = "factor"
    upscale: float = 1
    target_longest_side: int | None = None
    seed: int = 231
    device: str = "cuda"
    batch_size: int = 17
    temporal_overlap: int = 3
    color_correction: str = "lab"
    chunk_size: int = 170


class SeedVr2Engine:
    @property
    def python_path(self) -> Path:
        return ROOT / ".venv-seedvr2" / "Scripts" / "python.exe"

    @property
    def repo_path(self) -> Path:
        return ROOT / "external" / "SeedVR2_VideoUpscaler"

    @property
    def cli_path(self) -> Path:
        return self.repo_path / "inference_cli.py"

    @property
    def model_dir(self) -> Path:
        return ROOT / "models" / "seedvr2"

    @property
    def dit_model_path(self) -> Path:
        return self.model_dir / "seedvr2_ema_3b_fp8_e4m3fn.safetensors"

    @property
    def vae_model_path(self) -> Path:
        return self.model_dir / "ema_vae_fp16.safetensors"

    def status(self) -> dict:
        available = (
            self.python_path.exists()
            and self.cli_path.exists()
            and self.dit_model_path.exists()
            and self.vae_model_path.exists()
        )
        missing = []
        if not self.python_path.exists():
            missing.append(str(self.python_path))
        if not self.cli_path.exists():
            missing.append(str(self.cli_path))
        if not self.dit_model_path.exists():
            missing.append(str(self.dit_model_path))
        if not self.vae_model_path.exists():
            missing.append(str(self.vae_model_path))
        return {
            "available": available,
            "loaded": False,
            "repoPath": str(self.repo_path),
            "pythonPath": str(self.python_path),
            "modelDir": str(self.model_dir),
            "ditModelPresent": self.dit_model_path.exists(),
            "vaeModelPresent": self.vae_model_path.exists(),
            "missing": missing,
            "adapterSupport": "not_available",
            "adapterNote": (
                "Public SeedVR2 code exposes training configs, but no supported local "
                "video-to-adapter workflow comparable to the HYPIR LoRA adapter."
            ),
        }

    def _ensure_available(self) -> None:
        status = self.status()
        if not status["available"]:
            missing = ", ".join(status["missing"]) or "unknown SeedVR2 dependency"
            raise RuntimeError(f"SeedVR2 is not ready. Missing: {missing}")

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["TOKENIZERS_PARALLELISM"] = "false"
        return env

    def _source_size(self, input_path: Path) -> tuple[int, int]:
        suffix = input_path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
            with Image.open(input_path) as img:
                return img.width, img.height
        meta = probe_video(input_path)
        return int(meta["width"]), int(meta["height"])

    def _target_edges(self, input_path: Path, settings: SeedVr2Settings) -> tuple[int, int]:
        width, height = self._source_size(input_path)
        if width <= 0 or height <= 0:
            raise ValueError("Could not determine source dimensions for SeedVR2.")
        source_short = min(width, height)
        source_long = max(width, height)
        if settings.scale_by == "longest_side":
            if not settings.target_longest_side:
                raise ValueError("target_longest_side is required for SeedVR2 longest-side scaling.")
            target_long = int(settings.target_longest_side)
            target_short = max(64, round(target_long * source_short / source_long))
        else:
            factor = max(0.1, float(settings.upscale))
            target_short = max(64, round(source_short * factor))
            target_long = max(64, round(source_long * factor))
        return target_short, target_long

    def _run(
        self,
        input_path: Path,
        output_path: Path,
        settings: SeedVr2Settings,
        *,
        batch_size: int,
        chunk_size: int,
        load_cap: int | None = None,
        on_line: Callable[[str], None] | None = None,
        on_process: Callable[[subprocess.Popen], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        low_priority: bool = False,
    ) -> dict:
        self._ensure_available()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        seed = settings.seed if settings.seed != -1 else random.randint(0, 2**32 - 1)
        resolution, max_resolution = self._target_edges(input_path, settings)
        args = [
            str(self.python_path),
            str(self.cli_path),
            str(input_path),
            "--output",
            str(output_path),
            "--model_dir",
            str(self.model_dir),
            "--resolution",
            str(resolution),
            "--max_resolution",
            str(max_resolution),
            "--batch_size",
            str(batch_size),
            "--temporal_overlap",
            str(settings.temporal_overlap),
            "--video_backend",
            "ffmpeg",
            "--color_correction",
            settings.color_correction,
            "--seed",
            str(seed),
            "--cuda_device",
            "0",
        ]
        if chunk_size > 0:
            args.extend(["--chunk_size", str(chunk_size)])
        if load_cap is not None:
            args.extend(["--load_cap", str(load_cap)])

        process = subprocess.Popen(
            args,
            cwd=str(ROOT),
            env=self._env(),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0) if low_priority else 0,
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
                raise RuntimeError("SeedVR2 export cancelled")
        return_code = process.wait()
        if should_cancel and should_cancel():
            raise RuntimeError("SeedVR2 export cancelled")
        if return_code != 0:
            detail = "\n".join(output_tail[-10:])
            suffix = f"\nLast output:\n{detail}" if detail else ""
            raise RuntimeError(f"SeedVR2 failed with exit code {return_code}{suffix}")
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"SeedVR2 did not create output: {output_path}")
        return {"seed": seed, "width": None, "height": None}

    def enhance_file(
        self,
        input_path: Path,
        output_path: Path,
        settings: SeedVr2Settings,
        *,
        on_process: Callable[[subprocess.Popen], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        low_priority: bool = False,
    ) -> dict:
        result = self._run(
            input_path,
            output_path,
            settings,
            batch_size=1,
            chunk_size=0,
            load_cap=1,
            on_process=on_process,
            should_cancel=should_cancel,
            low_priority=low_priority,
        )
        with Image.open(output_path) as img:
            result["width"] = img.width
            result["height"] = img.height
        return result

    def enhance_video(
        self,
        input_path: Path,
        output_path: Path,
        settings: SeedVr2Settings,
        *,
        on_line: Callable[[str], None] | None = None,
        on_process: Callable[[subprocess.Popen], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        low_priority: bool = False,
    ) -> dict:
        return self._run(
            input_path,
            output_path,
            settings,
            batch_size=settings.batch_size,
            chunk_size=settings.chunk_size,
            on_line=on_line,
            on_process=on_process,
            should_cancel=should_cancel,
            low_priority=low_priority,
        )


engine = SeedVr2Engine()
