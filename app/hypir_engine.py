from __future__ import annotations

import os
import random
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
HYPIR_ROOT = ROOT / "external" / "HYPIR"
if str(HYPIR_ROOT) not in sys.path:
    sys.path.insert(0, str(HYPIR_ROOT))

from HYPIR.enhancer.sd2 import SD2Enhancer  # noqa: E402


LORA_MODULES = [
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
    "conv",
    "conv1",
    "conv2",
    "conv_shortcut",
    "conv_out",
    "proj_in",
    "proj_out",
    "ff.net.2",
    "ff.net.0.proj",
]


@dataclass(frozen=True)
class HypirSettings:
    prompt: str = ""
    scale_by: str = "factor"
    upscale: float = 1
    target_longest_side: int | None = None
    patch_size: int = 512
    stride: int = 256
    seed: int = 231
    device: str = "cuda"


class HypirEngine:
    def __init__(self) -> None:
        self._model: SD2Enhancer | None = None
        self._device: str | None = None
        self._lock = threading.Lock()
        self._to_tensor = transforms.ToTensor()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def status(self) -> dict:
        return {
            "loaded": self.loaded,
            "device": self._device,
            "cudaAvailable": torch.cuda.is_available(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "weightPath": str(self.weight_path),
            "baseModelPath": str(self.base_model_path),
            "weightPresent": self.weight_path.exists(),
            "baseModelPresent": self.base_model_path.exists(),
        }

    @property
    def weight_path(self) -> Path:
        return ROOT / "models" / "hypir" / "HYPIR_sd2.pth"

    @property
    def base_model_path(self) -> Path:
        return ROOT / "models" / "stable-diffusion-2-1-base"

    def _resolve_device(self, requested_device: str) -> str:
        if requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. HYPIR needs a CUDA GPU for practical local video export.")
            return "cuda"
        return requested_device

    def _load_locked(self, device: str) -> None:
        if self._model is not None and self._device == device:
            return
        if not self.weight_path.exists():
            raise FileNotFoundError(f"Missing HYPIR weights: {self.weight_path}")
        if not self.base_model_path.exists():
            raise FileNotFoundError(f"Missing Stable Diffusion base model: {self.base_model_path}")

        model = SD2Enhancer(
            base_model_path=str(self.base_model_path),
            weight_path=str(self.weight_path),
            lora_modules=LORA_MODULES,
            lora_rank=256,
            model_t=200,
            coeff_t=200,
            device=device,
        )
        model.init_models()
        self._model = model
        self._device = device

    def enhance_file(self, input_path: Path, output_path: Path, settings: HypirSettings) -> dict:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        seed = settings.seed if settings.seed != -1 else random.randint(0, 2**32 - 1)
        scale_by = settings.scale_by
        target_longest_side = settings.target_longest_side
        if scale_by == "longest_side" and not target_longest_side:
            raise ValueError("target_longest_side is required when scale_by is longest_side")

        with self._lock:
            device = self._resolve_device(settings.device)
            self._load_locked(device)
            assert self._model is not None

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            image = Image.open(input_path).convert("RGB")
            model_upscale = settings.upscale
            if scale_by == "factor" and 0 < settings.upscale < 1:
                image = image.resize(
                    (
                        max(1, round(image.width * settings.upscale)),
                        max(1, round(image.height * settings.upscale)),
                    ),
                    Image.Resampling.LANCZOS,
                )
                model_upscale = 1
            tensor = self._to_tensor(image).unsqueeze(0)
            result = self._model.enhance(
                lq=tensor,
                prompt=settings.prompt,
                scale_by=scale_by,
                upscale=model_upscale,
                target_longest_side=target_longest_side,
                patch_size=settings.patch_size,
                stride=settings.stride,
                return_type="pil",
            )[0]
            result.save(output_path)
            out_size = result.size
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {"seed": seed, "width": out_size[0], "height": out_size[1]}


engine = HypirEngine()
