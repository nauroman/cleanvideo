from __future__ import annotations

import os
import random
import re
import shutil
import subprocess
import uuid
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

from .resource_control import process_creationflags


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SupirSettings:
    scale_by: str = "factor"
    upscale: int = 1
    target_longest_side: int | None = None
    seed: int = 231
    sign: str = "Q"
    min_size: int = 1024
    edm_steps: int = 50
    s_stage1: int = -1
    s_churn: int = 5
    s_noise: float = 1.01
    s_cfg: float = 4.0
    s_stage2: float = 1.0
    color_fix_type: str = "Wavelet"
    ae_dtype: str = "bf16"
    diff_dtype: str = "fp16"
    no_llava: bool = True
    loading_half_params: bool = False
    use_tile_vae: bool = True
    encoder_tile_size: int = 512
    decoder_tile_size: int = 64


class SupirEngine:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else ROOT

    @property
    def python_path(self) -> Path:
        return self.root / ".venv-supir" / "Scripts" / "python.exe"

    @property
    def repo_path(self) -> Path:
        return self.root / "external" / "SUPIR"

    @property
    def cli_path(self) -> Path:
        return self.root / "scripts" / "supir_cli.py"

    @property
    def config_path(self) -> Path:
        return self.repo_path / "options" / "SUPIR_v0.yaml"

    @property
    def checkpoint_path_config(self) -> Path:
        return self.repo_path / "CKPT_PTH.py"

    def _configured_checkpoint_paths(self) -> dict[str, Path]:
        if not self.config_path.exists():
            return {}
        result: dict[str, Path] = {}
        pattern = re.compile(r"^(SDXL_CKPT|SUPIR_CKPT_Q|SUPIR_CKPT_F):\s*(.+?)\s*$")
        for line in self.config_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = pattern.match(line)
            if not match:
                continue
            key, raw_value = match.groups()
            value = raw_value.strip().strip("'\"")
            if not value or value == "~":
                continue
            path = Path(value)
            if not path.is_absolute():
                path = self.repo_path / path
            result[key] = path
        return result

    def _configured_model_paths(self) -> dict[str, Path]:
        if not self.checkpoint_path_config.exists():
            return {}
        result: dict[str, Path] = {}
        keys = {"SDXL_CLIP1_PATH", "SDXL_CLIP2_CKPT_PTH", "LLAVA_CLIP_PATH", "LLAVA_MODEL_PATH"}
        try:
            tree = ast.parse(self.checkpoint_path_config.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            return result
        for node in tree.body:
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
                continue
            if not isinstance(node.value.value, str):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Name) or target.id not in keys:
                    continue
                path = Path(node.value.value)
                if not path.is_absolute():
                    path = self.repo_path / path
                result[target.id] = path
        return result

    def status(self) -> dict:
        missing = []
        for path in [self.python_path, self.cli_path, self.config_path, self.checkpoint_path_config]:
            if not path.exists():
                missing.append(str(path))
        configured_checkpoints = self._configured_checkpoint_paths()
        configured_model_paths = self._configured_model_paths()
        required_checkpoint_keys = {"SDXL_CKPT", "SUPIR_CKPT_Q"}
        required_model_path_keys = {"SDXL_CLIP1_PATH", "SDXL_CLIP2_CKPT_PTH"}
        optional_missing = []
        if self.config_path.exists():
            for key in sorted(required_checkpoint_keys - configured_checkpoints.keys()):
                missing.append(f"{key}: not configured")
        if self.checkpoint_path_config.exists():
            for key in sorted(required_model_path_keys - configured_model_paths.keys()):
                missing.append(f"{key}: not configured")
        for key, path in configured_checkpoints.items():
            if not path.exists():
                if key in required_checkpoint_keys:
                    missing.append(f"{key}: {path}")
                else:
                    optional_missing.append(f"{key}: {path}")
        for key, path in configured_model_paths.items():
            if not path.exists():
                if key in required_model_path_keys:
                    missing.append(f"{key}: {path}")
                else:
                    optional_missing.append(f"{key}: {path}")
        available = not missing
        return {
            "available": available,
            "loaded": False,
            "repoPath": str(self.repo_path),
            "pythonPath": str(self.python_path),
            "scriptPath": str(self.cli_path),
            "configPath": str(self.config_path),
            "checkpointPathConfig": str(self.checkpoint_path_config),
            "configuredCheckpoints": {key: str(path) for key, path in configured_checkpoints.items()},
            "configuredModelPaths": {key: str(path) for key, path in configured_model_paths.items()},
            "missing": missing,
            "optionalMissing": optional_missing,
            "adapterSupport": "not_available",
            "adapterNote": (
                "SUPIR is exposed here as a per-frame folder inference engine. Film adapters "
                "remain HYPIR-only."
            ),
            "blockedReason": None if available else "SUPIR runtime, config, or checkpoint files are missing.",
        }

    def _ensure_available(self) -> None:
        status = self.status()
        if not status["available"]:
            missing = ", ".join(status["missing"]) or "unknown SUPIR dependency"
            raise RuntimeError(f"SUPIR is not ready. Missing: {missing}")

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["TOKENIZERS_PARALLELISM"] = "false"
        return env

    def output_for_input(self, output_dir: Path, input_path: Path) -> Path:
        return output_dir / f"{input_path.stem}_0.png"

    def build_command(self, input_dir: Path, output_dir: Path, settings: SupirSettings) -> list[str]:
        seed = settings.seed if settings.seed != -1 else random.randint(0, 2**32 - 1)
        args = [
            str(self.python_path),
            str(self.cli_path),
            "--img_dir",
            str(input_dir),
            "--save_dir",
            str(output_dir),
            "--upscale",
            str(max(1, min(4, int(settings.upscale or 1)))),
            "--SUPIR_sign",
            settings.sign,
            "--seed",
            str(seed),
            "--min_size",
            str(settings.min_size),
            "--edm_steps",
            str(settings.edm_steps),
            "--s_stage1",
            str(settings.s_stage1),
            "--s_churn",
            str(settings.s_churn),
            "--s_noise",
            str(settings.s_noise),
            "--s_cfg",
            str(settings.s_cfg),
            "--s_stage2",
            str(settings.s_stage2),
            "--num_samples",
            "1",
            "--color_fix_type",
            settings.color_fix_type,
            "--ae_dtype",
            settings.ae_dtype,
            "--diff_dtype",
            settings.diff_dtype,
        ]
        if settings.no_llava:
            args.append("--no_llava")
        if settings.loading_half_params:
            args.append("--loading_half_params")
        if settings.use_tile_vae:
            args.extend(
                [
                    "--use_tile_vae",
                    "--encoder_tile_size",
                    str(settings.encoder_tile_size),
                    "--decoder_tile_size",
                    str(settings.decoder_tile_size),
                ]
            )
        return args

    def enhance_frames(
        self,
        input_dir: Path,
        output_dir: Path,
        settings: SupirSettings,
        *,
        on_line: Callable[[str], None] | None = None,
        on_process: Callable[[subprocess.Popen], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        low_priority: bool = False,
    ) -> dict:
        self._ensure_available()
        output_dir.mkdir(parents=True, exist_ok=True)
        args = self.build_command(input_dir, output_dir, settings)
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
                raise RuntimeError("SUPIR export cancelled")
        return_code = process.wait()
        if should_cancel and should_cancel():
            raise RuntimeError("SUPIR export cancelled")
        if return_code != 0:
            detail = "\n".join(output_tail[-10:])
            suffix = f"\nLast output:\n{detail}" if detail else ""
            raise RuntimeError(f"SUPIR failed with exit code {return_code}{suffix}")
        return {"seed": settings.seed}

    def enhance_file(
        self,
        input_path: Path,
        output_path: Path,
        settings: SupirSettings,
        *,
        on_process: Callable[[subprocess.Popen], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        low_priority: bool = False,
    ) -> dict:
        self._ensure_available()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        work_dir = output_path.parent / f"{output_path.stem}_supir_{uuid.uuid4().hex[:8]}"
        input_dir = work_dir / "input"
        raw_output_dir = work_dir / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        raw_output_dir.mkdir(parents=True, exist_ok=True)
        isolated = input_dir / f"input{input_path.suffix.lower() or '.png'}"
        try:
            shutil.copy2(input_path, isolated)
            result = self.enhance_frames(
                input_dir,
                raw_output_dir,
                settings,
                on_process=on_process,
                should_cancel=should_cancel,
                low_priority=low_priority,
            )
            generated = self.output_for_input(raw_output_dir, isolated)
            if not generated.exists() or generated.stat().st_size == 0:
                raise RuntimeError(f"SUPIR did not create output: {generated}")
            shutil.copy2(generated, output_path)
            with Image.open(output_path) as img:
                result["width"] = img.width
                result["height"] = img.height
            return result
        finally:
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)


engine = SupirEngine()
