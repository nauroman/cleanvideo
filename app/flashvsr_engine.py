from __future__ import annotations

import os
import random
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FlashVsrSettings:
    scale_by: str = "factor"
    upscale: float = 4
    target_longest_side: int | None = None
    seed: int = 231
    variant: str = "tiny_long"
    sparse_ratio: float = 2.0
    local_range: int = 11
    quality: int = 6


FLASHVSR_TRANSIENT_IMPORT_ATTEMPTS = 3


def is_transient_import_corruption(output: str) -> bool:
    normalized = output.replace("\\", "/")
    return (
        "TypeError: unsupported operand type(s) for +:" in normalized
        and "Tokenizer" in normalized
        and "re/_parser.py" in normalized
    )


def is_transient_torch_load_corruption(output: str) -> bool:
    normalized = output.replace("\\", "/")
    return (
        "torch/serialization.py" in normalized
        and "torch/_weights_only_unpickler.py" in normalized
        and "TypeError: 'str' object is not callable" in normalized
    )


def is_transient_flashvsr_worker_error(output: str) -> bool:
    return is_transient_import_corruption(output) or is_transient_torch_load_corruption(output)


class FlashVsrEngine:
    def __init__(self) -> None:
        self._status_cache: tuple[float, dict] | None = None
        self._status_cache_ttl = 10.0

    @property
    def python_path(self) -> Path:
        return ROOT / ".venv-flashvsr" / "Scripts" / "python.exe"

    @property
    def wsl_python_path(self) -> str:
        return "/home/user/.cleanvideo/flashvsr-wsl/.venv/bin/python"

    @property
    def wsl_cuda_path(self) -> str:
        return "/home/user/.cleanvideo/cuda-12.4"

    @property
    def repo_path(self) -> Path:
        return ROOT / "external" / "FlashVSR"

    @property
    def runner_path(self) -> Path:
        return ROOT / "scripts" / "flashvsr_cli.py"

    @property
    def wsl_python_runner_path(self) -> Path:
        return ROOT / "scripts" / "flashvsr_wsl_python.sh"

    @property
    def model_dir(self) -> Path:
        return self.repo_path / "examples" / "WanVSR" / "FlashVSR-v1.1"

    @property
    def required_models(self) -> list[Path]:
        return [
            self.model_dir / "diffusion_pytorch_model_streaming_dmd.safetensors",
            self.model_dir / "LQ_proj_in.ckpt",
            self.model_dir / "TCDecoder.ckpt",
            self.model_dir / "Wan2.1_VAE.pth",
        ]

    def _probe_code(self, *, import_block_sparse: bool) -> str:
        block_sparse_probe = [
            "block_sparse = importlib.util.find_spec('block_sparse_attn') is not None",
            "block_sparse_cuda = importlib.util.find_spec('block_sparse_attn_cuda') is not None",
            "import_error = None",
        ]
        if import_block_sparse:
            block_sparse_probe.extend(
                [
                    "try:",
                    "    import block_sparse_attn, block_sparse_attn_cuda",
                    "except Exception as exc:",
                    "    import_error = str(exc)",
                    "else:",
                    "    block_sparse = True",
                    "    block_sparse_cuda = True",
                ]
            )
        return (
            "import importlib.util, json, torch\n"
            + "\n".join(block_sparse_probe)
            + "\nprint(json.dumps({"
            "'torch': torch.__version__, "
            "'cuda': torch.version.cuda, "
            "'cudaAvailable': torch.cuda.is_available(), "
            "'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "
            "'diffsynth': importlib.util.find_spec('diffsynth') is not None, "
            "'blockSparse': block_sparse, "
            "'blockSparseCuda': block_sparse_cuda, "
            "'blockSparseImportError': import_error"
            "}))\n"
        )

    def _windows_python_probe(self) -> dict:
        if not self.python_path.exists():
            return {"ok": False, "error": f"Missing Python env: {self.python_path}"}
        try:
            proc = subprocess.run(
                [str(self.python_path), "-c", self._probe_code(import_block_sparse=False)],
                cwd=str(ROOT),
                text=True,
                capture_output=True,
                timeout=15,
                check=True,
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        try:
            import json

            return {"ok": True, **json.loads(proc.stdout.strip())}
        except Exception as exc:
            return {"ok": False, "error": f"{exc}: {proc.stdout.strip()}"}

    def _wslpath(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            proc = subprocess.run(
                ["wsl.exe", "wslpath", "-a", str(resolved)],
                text=True,
                capture_output=True,
                timeout=10,
                check=True,
            )
            return proc.stdout.strip()
        except Exception:
            drive = resolved.drive.rstrip(":").lower()
            rest = resolved.as_posix().split(":", 1)[-1]
            return f"/mnt/{drive}{rest}" if drive else resolved.as_posix()

    def _wsl_runner_args(self, *args: str) -> list[str]:
        return ["wsl.exe", "bash", self._wslpath(self.wsl_python_runner_path), *args]

    def _wsl_python_probe(self) -> dict:
        if not self.wsl_python_runner_path.exists():
            return {"ok": False, "error": f"Missing WSL runner: {self.wsl_python_runner_path}"}
        try:
            proc = subprocess.run(
                self._wsl_runner_args("-c", self._probe_code(import_block_sparse=True)),
                cwd=str(ROOT),
                text=True,
                capture_output=True,
                timeout=30,
                check=True,
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        try:
            import json

            lines = [line for line in proc.stdout.splitlines() if line.strip().startswith("{")]
            return {"ok": True, **json.loads(lines[-1] if lines else proc.stdout.strip())}
        except Exception as exc:
            return {"ok": False, "error": f"{exc}: {proc.stdout.strip()}"}

    def status(self) -> dict:
        now = time.monotonic()
        if self._status_cache and now - self._status_cache[0] < self._status_cache_ttl:
            return dict(self._status_cache[1])

        windows_probe = self._windows_python_probe()
        wsl_probe = self._wsl_python_probe()
        models_present = all(path.exists() for path in self.required_models)
        windows_available = (
            self.python_path.exists()
            and self.repo_path.exists()
            and self.runner_path.exists()
            and models_present
            and bool(windows_probe.get("ok"))
            and bool(windows_probe.get("blockSparse"))
            and bool(windows_probe.get("blockSparseCuda"))
        )
        wsl_available = (
            self.wsl_python_runner_path.exists()
            and self.repo_path.exists()
            and self.runner_path.exists()
            and models_present
            and bool(wsl_probe.get("ok"))
            and bool(wsl_probe.get("blockSparse"))
            and bool(wsl_probe.get("blockSparseCuda"))
        )
        probe = wsl_probe if wsl_available or not windows_available else windows_probe
        backend = "windows" if windows_available else "wsl" if wsl_available else "none"
        missing = []
        if not self.repo_path.exists():
            missing.append(str(self.repo_path))
        if not self.runner_path.exists():
            missing.append(str(self.runner_path))
        for model_path in self.required_models:
            if not model_path.exists():
                missing.append(str(model_path))
        if not windows_available and not wsl_available:
            if not self.python_path.exists() and not self.wsl_python_runner_path.exists():
                missing.append(str(self.python_path))
                missing.append(str(self.wsl_python_runner_path))
            if windows_probe.get("ok") and not windows_probe.get("blockSparse"):
                missing.append("windows:block_sparse_attn")
            if wsl_probe.get("ok") and not wsl_probe.get("blockSparse"):
                missing.append("wsl:block_sparse_attn")
            if wsl_probe.get("ok") and wsl_probe.get("blockSparseImportError"):
                missing.append(f"wsl:{wsl_probe['blockSparseImportError']}")
            if not windows_probe.get("ok") and not wsl_probe.get("ok"):
                missing.append(wsl_probe.get("error") or windows_probe.get("error") or "FlashVSR Python probe failed")

        available = windows_available or wsl_available
        result = {
            "available": available,
            "loaded": False,
            "backend": backend,
            "repoPath": str(self.repo_path),
            "pythonPath": str(self.python_path),
            "wslPythonPath": self.wsl_python_path,
            "wslCudaPath": self.wsl_cuda_path,
            "wslRunnerPath": str(self.wsl_python_runner_path),
            "modelDir": str(self.model_dir),
            "modelsPresent": models_present,
            "diffsynthPresent": bool(probe.get("diffsynth")),
            "blockSparsePresent": bool(probe.get("blockSparse")),
            "blockSparseCudaPresent": bool(probe.get("blockSparseCuda")),
            "torch": probe.get("torch"),
            "cuda": probe.get("cuda"),
            "cudaAvailable": probe.get("cudaAvailable"),
            "gpu": probe.get("gpu"),
            "missing": missing,
            "adapterSupport": "not_available",
            "adapterNote": (
                "FlashVSR publishes inference code and describes its training pipeline/dataset, "
                "but there is no released local video-to-adapter workflow for a custom film adapter."
            ),
            "blockedReason": (
                None
                if available
                else "Official FlashVSR requires block_sparse_attn/LCSA. Windows failed, and WSL bridge is not ready."
            ),
        }
        self._status_cache = (time.monotonic(), result)
        return dict(result)

    def _active_backend(self) -> str:
        status = self.status()
        if not status["available"]:
            missing = ", ".join(status["missing"]) or status.get("blockedReason") or "unknown dependency"
            raise RuntimeError(f"FlashVSR is not ready. Missing or blocked: {missing}")
        return str(status.get("backend") or "windows")

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["TOKENIZERS_PARALLELISM"] = "false"
        return env

    def _scale(self, settings: FlashVsrSettings, source_longest_side: int | None = None) -> float:
        if settings.scale_by == "longest_side" and settings.target_longest_side and source_longest_side:
            return max(1.0, min(4.0, settings.target_longest_side / source_longest_side))
        return max(1.0, min(4.0, float(settings.upscale or 4.0)))

    def enhance_video(
        self,
        input_path: Path,
        output_path: Path,
        settings: FlashVsrSettings,
        *,
        source_longest_side: int | None = None,
        on_line: Callable[[str], None] | None = None,
        on_process: Callable[[subprocess.Popen], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        low_priority: bool = False,
        timeout_seconds: float | None = None,
        total_frames: int | None = None,
        fps: float | None = None,
        streaming: bool = False,
        live_original_path: Path | None = None,
        live_enhanced_path: Path | None = None,
    ) -> dict:
        backend = self._active_backend()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        seed = settings.seed if settings.seed != -1 else random.randint(0, 2**32 - 1)
        if backend == "wsl":
            args = self._wsl_runner_args(
                self._wslpath(self.runner_path),
                self._wslpath(input_path),
                "--output",
                self._wslpath(output_path),
                "--variant",
                settings.variant,
                "--scale",
                f"{self._scale(settings, source_longest_side):.4f}",
                "--seed",
                str(seed),
                "--sparse_ratio",
                str(settings.sparse_ratio),
                "--local_range",
                str(settings.local_range),
                "--quality",
                str(settings.quality),
            )
            if total_frames is not None:
                args.extend(["--total_frames", str(max(1, int(total_frames)))])
            if fps is not None and fps > 0:
                args.extend(["--fps", str(float(fps))])
            if streaming:
                args.append("--streaming")
            if live_original_path is not None:
                args.extend(["--live_original", self._wslpath(live_original_path)])
            if live_enhanced_path is not None:
                args.extend(["--live_enhanced", self._wslpath(live_enhanced_path)])
            env = os.environ.copy()
        else:
            args = [
                str(self.python_path),
                str(self.runner_path),
                str(input_path),
                "--output",
                str(output_path),
                "--variant",
                settings.variant,
                "--scale",
                f"{self._scale(settings, source_longest_side):.4f}",
                "--seed",
                str(seed),
                "--sparse_ratio",
                str(settings.sparse_ratio),
                "--local_range",
                str(settings.local_range),
                "--quality",
                str(settings.quality),
            ]
            if total_frames is not None:
                args.extend(["--total_frames", str(max(1, int(total_frames)))])
            if fps is not None and fps > 0:
                args.extend(["--fps", str(float(fps))])
            if streaming:
                args.append("--streaming")
            if live_original_path is not None:
                args.extend(["--live_original", str(live_original_path)])
            if live_enhanced_path is not None:
                args.extend(["--live_enhanced", str(live_enhanced_path)])
            env = self._env()
        for attempt in range(1, FLASHVSR_TRANSIENT_IMPORT_ATTEMPTS + 1):
            if attempt > 1:
                output_path.unlink(missing_ok=True)
            process = subprocess.Popen(
                args,
                cwd=str(ROOT),
                env=env,
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
            timed_out = threading.Event()
            timeout_timer: threading.Timer | None = None

            def expire_process() -> None:
                if process.poll() is not None:
                    return
                timed_out.set()
                try:
                    process.terminate()
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                except Exception:
                    pass

            if timeout_seconds is not None and timeout_seconds > 0:
                timeout_timer = threading.Timer(timeout_seconds, expire_process)
                timeout_timer.daemon = True
                timeout_timer.start()

            try:
                for line in process.stdout:
                    stripped = line.rstrip()
                    if stripped:
                        output_tail.append(stripped)
                        output_tail = output_tail[-20:]
                    if stripped.startswith("wsl: Failed to mount "):
                        continue
                    if on_line:
                        on_line(stripped)
                    if should_cancel and should_cancel():
                        process.terminate()
                        try:
                            process.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        raise RuntimeError("FlashVSR export cancelled")
                return_code = process.wait()
            finally:
                if timeout_timer is not None:
                    timeout_timer.cancel()
            if timed_out.is_set():
                raise RuntimeError(f"FlashVSR timed out after {int(timeout_seconds or 0)} seconds")
            if should_cancel and should_cancel():
                raise RuntimeError("FlashVSR export cancelled")
            if return_code == 0:
                break

            detail = "\n".join(output_tail[-10:])
            if attempt < FLASHVSR_TRANSIENT_IMPORT_ATTEMPTS and is_transient_flashvsr_worker_error(detail):
                if on_line:
                    on_line(
                        "FlashVSR hit transient Python runtime corruption; "
                        f"retrying ({attempt + 1}/{FLASHVSR_TRANSIENT_IMPORT_ATTEMPTS})"
                    )
                time.sleep(1.0)
                continue

            suffix = f"\nLast output:\n{detail}" if detail else ""
            if return_code == 9:
                raise RuntimeError(
                    "FlashVSR worker was killed with exit code 9. This usually means the selected "
                    "native resolution is above the available WSL RAM/VRAM budget. Lower Resolution/Scale Mode."
                    f"{suffix}"
                )
            raise RuntimeError(f"FlashVSR failed with exit code {return_code}{suffix}")
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"FlashVSR did not create output: {output_path}")
        return {"seed": seed, "width": None, "height": None}


engine = FlashVsrEngine()
