from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
WAN_DIR = ROOT / "external" / "FlashVSR" / "examples" / "WanVSR"


def load_official_module(variant: str):
    script_name = {
        "tiny_long": "infer_flashvsr_v1.1_tiny_long_video.py",
        "tiny": "infer_flashvsr_v1.1_tiny.py",
        "full": "infer_flashvsr_v1.1_full.py",
    }[variant]
    script_path = WAN_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing official FlashVSR script: {script_path}")
    if str(WAN_DIR) not in sys.path:
        sys.path.insert(0, str(WAN_DIR))
    spec = importlib.util.spec_from_file_location("cleanvideo_flashvsr_official", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import official FlashVSR script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def error_looks_like_transformers_cache_error(error: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = str(current).lower()
        if (
            "unknown opcode" in text
            or "transformers.models.t5.modeling_t5" in text
            or "object does not support the context manager protocol" in text
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def clear_transformers_bytecode_cache() -> None:
    try:
        import transformers
    except Exception:
        return

    package_root = Path(transformers.__file__).resolve().parent
    roots = [
        package_root / "models" / "t5",
        package_root / "utils",
    ]
    removed = 0
    for root in roots:
        if not root.exists():
            continue
        for cache_dir in root.rglob("__pycache__"):
            shutil.rmtree(cache_dir, ignore_errors=True)
            removed += 1
    importlib.invalidate_caches()
    if removed:
        print(f"Cleared transformers bytecode cache directories: {removed}", flush=True)


def reset_transformers_import_cache() -> None:
    clear_transformers_bytecode_cache()
    for module_name in list(sys.modules):
        if module_name.startswith("transformers.models.t5"):
            sys.modules.pop(module_name, None)
    importlib.invalidate_caches()


def preload_full_variant_dependencies() -> None:
    """Preload T5 dependencies used by FlashVSR Full and recover stale pyc caches."""
    try:
        import importlib

        importlib.import_module("transformers.models.t5.modeling_t5")
        from transformers import T5EncoderModel  # noqa: F401
    except RuntimeError as exc:
        if not error_looks_like_transformers_cache_error(exc):
            raise
        reset_transformers_import_cache()
        importlib.import_module("transformers.models.t5.modeling_t5")
        from transformers import T5EncoderModel  # noqa: F401


def patch_full_variant_float_scale(module) -> None:
    """FlashVSR full v1.1 annotates scale as int but our UI can pass float scales."""

    def scaled_size(width: int, height: int, scale: float) -> tuple[int, int]:
        if width <= 0 or height <= 0:
            raise ValueError("invalid original size")
        if scale <= 0:
            raise ValueError("scale must be > 0")
        return max(1, int(round(width * scale))), max(1, int(round(height * scale)))

    def compute_scaled_and_target_dims(w0: int, h0: int, scale: float = 4.0, multiple: int = 128):
        sW, sH = scaled_size(w0, h0, float(scale))
        tW = max(multiple, (sW // multiple) * multiple)
        tH = max(multiple, (sH // multiple) * multiple)
        return sW, sH, tW, tH

    def upscale_then_center_crop(img, scale: float, tW: int, tH: int):
        sW, sH = scaled_size(img.width, img.height, float(scale))
        up = img.resize((sW, sH), module.Image.BICUBIC)
        left = max(0, (sW - tW) // 2)
        top = max(0, (sH - tH) // 2)
        return up.crop((left, top, left + tW, top + tH))

    module.compute_scaled_and_target_dims = compute_scaled_and_target_dims
    module.upscale_then_center_crop = upscale_then_center_crop


def main() -> int:
    parser = argparse.ArgumentParser(description="CleanVideo adapter for official FlashVSR v1.1.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["tiny_long", "tiny", "full"], default="tiny_long")
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--sparse_ratio", type=float, default=2.0)
    parser.add_argument("--local_range", type=int, default=11)
    parser.add_argument("--quality", type=int, default=6)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("FlashVSR needs CUDA.")
    if not (WAN_DIR / "FlashVSR-v1.1" / "diffusion_pytorch_model_streaming_dmd.safetensors").exists():
        raise FileNotFoundError("Missing FlashVSR-v1.1 model weights.")

    if args.variant == "full":
        preload_full_variant_dependencies()

    try:
        module = load_official_module(args.variant)
    except RuntimeError as exc:
        if not error_looks_like_transformers_cache_error(exc):
            raise
        reset_transformers_import_cache()
        if args.variant == "full":
            preload_full_variant_dependencies()
        module = load_official_module(args.variant)
    if args.variant == "full":
        patch_full_variant_float_scale(module)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    os.chdir(WAN_DIR)
    print(torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))
    try:
        pipe = module.init_pipeline()
    except RuntimeError as exc:
        if not error_looks_like_transformers_cache_error(exc):
            raise
        reset_transformers_import_cache()
        if args.variant == "full":
            preload_full_variant_dependencies()
        pipe = module.init_pipeline()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    lq_video, height, width, frame_count, fps = module.prepare_input_tensor(
        str(args.input.resolve()),
        scale=args.scale,
        dtype=torch.bfloat16,
        device="cuda",
    )
    video = pipe(
        prompt="",
        negative_prompt="",
        cfg_scale=1.0,
        num_inference_steps=1,
        seed=args.seed,
        LQ_video=lq_video,
        num_frames=frame_count,
        height=height,
        width=width,
        is_full_block=False,
        if_buffer=True,
        topk_ratio=args.sparse_ratio * 768 * 1280 / (height * width),
        kv_ratio=3.0,
        local_range=args.local_range,
        color_fix=True,
    )
    frames = module.tensor2video(video)
    module.save_video(frames, str(output_path), fps=fps, quality=args.quality)
    print(f"Done: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
