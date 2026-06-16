from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image


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


def model_output_frame_count(source_frames: int) -> int:
    """FlashVSR emits 8n+5 frames; pad with cloned tail frames and trim later."""
    source_frames = max(1, int(source_frames))
    remainder = source_frames % 8
    if remainder <= 5:
        padded = source_frames + (5 - remainder)
    else:
        padded = source_frames + (13 - remainder)
    return max(21, padded)


def count_reader_frames(reader, meta: dict) -> int:
    try:
        frame_count = meta.get("nframes")
        if isinstance(frame_count, int) and frame_count > 0:
            return frame_count
    except Exception:
        pass
    try:
        return int(reader.count_frames())
    except Exception:
        count = 0
        try:
            while True:
                reader.get_data(count)
                count += 1
        except Exception:
            return count


class StreamingLqFrameBuffer:
    def __init__(
        self,
        module,
        video_path: Path,
        *,
        scale: float,
        dtype: torch.dtype,
        expected_frames: int | None = None,
        fps_override: float | None = None,
    ) -> None:
        self.module = module
        self.video_path = video_path
        self.scale = scale
        self.dtype = dtype
        self.reader = imageio.get_reader(str(video_path))
        self.meta = {}
        try:
            self.meta = dict(self.reader.get_meta_data())
        except Exception:
            self.meta = {}

        first = Image.fromarray(self.reader.get_data(0)).convert("RGB")
        self.last_image = first
        self.source_width, self.source_height = first.size
        self.source_frames = max(1, int(expected_frames or count_reader_frames(self.reader, self.meta) or 1))
        fps_value = fps_override if fps_override and fps_override > 0 else self.meta.get("fps", 30)
        self.fps = float(fps_value) if isinstance(fps_value, (int, float)) else 30.0

        _scaled_width, _scaled_height, self.target_width, self.target_height = module.compute_scaled_and_target_dims(
            self.source_width,
            self.source_height,
            scale=scale,
            multiple=128,
        )
        self.output_frames = model_output_frame_count(self.source_frames)
        self.model_frames = self.output_frames + 4
        self.base_index = 0
        self.cursor = 0
        self.frames: list[torch.Tensor] = []

        print(
            f"[{video_path.name}] Original Resolution: {self.source_width}x{self.source_height} | "
            f"Original Frames: {self.source_frames} | FPS: {self.fps:g}",
            flush=True,
        )
        print(
            f"[{video_path.name}] Streaming target: {self.target_width}x{self.target_height} | "
            f"Model Frames: {self.model_frames} | Output Frames: {self.output_frames}",
            flush=True,
        )

    def close(self) -> None:
        try:
            self.reader.close()
        except Exception:
            pass

    def _read_processed_frame(self, frame_index: int) -> torch.Tensor:
        if frame_index < self.source_frames:
            image = Image.fromarray(self.reader.get_data(frame_index)).convert("RGB")
            self.last_image = image
        else:
            image = self.last_image
        image_out = self.module.upscale_then_center_crop(
            image,
            scale=self.scale,
            tW=self.target_width,
            tH=self.target_height,
        )
        return self.module.pil_to_tensor_neg1_1(image_out, self.dtype, "cpu")

    def ensure(self, end_index: int) -> None:
        while self.base_index + len(self.frames) < end_index:
            self.frames.append(self._read_processed_frame(self.cursor))
            self.cursor += 1

    def slice(self, start_index: int, end_index: int) -> torch.Tensor:
        if end_index <= start_index:
            raise ValueError(f"Invalid frame slice {start_index}:{end_index}")
        self.ensure(end_index)
        offset = start_index - self.base_index
        if offset < 0:
            raise RuntimeError(f"Requested discarded FlashVSR frame slice {start_index}:{end_index}")
        selected = self.frames[offset : offset + (end_index - start_index)]
        return torch.stack(selected, 0).permute(1, 0, 2, 3).unsqueeze(0)

    def discard_before(self, frame_index: int) -> None:
        drop = max(0, min(frame_index - self.base_index, len(self.frames)))
        if drop:
            del self.frames[:drop]
            self.base_index += drop


def tensor_video_to_uint8(frames: torch.Tensor) -> np.ndarray:
    frames = frames.detach().float().permute(1, 2, 3, 0)
    return ((frames + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)


def save_live_frame_pair(
    original_path: Path,
    enhanced_path: Path,
    original_frame: np.ndarray,
    enhanced_frame: np.ndarray,
    quality: int,
) -> None:
    original_path.parent.mkdir(parents=True, exist_ok=True)
    enhanced_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"quality": quality, "subsampling": 0}
    Image.fromarray(original_frame).save(original_path, format="JPEG", **save_kwargs)
    Image.fromarray(enhanced_frame).save(enhanced_path, format="JPEG", **save_kwargs)


def append_video_frames(
    writer,
    frames: torch.Tensor,
    original_frames: torch.Tensor,
    remaining: int,
    total_frames: int,
    written: int,
    *,
    fps: float,
    live_original: Path | None,
    live_enhanced: Path | None,
    live_quality: int,
) -> int:
    frames = tensor_video_to_uint8(frames)
    original_frames = tensor_video_to_uint8(original_frames)
    frame_count = min(len(frames), remaining)
    for local_index, frame in enumerate(frames[:frame_count]):
        writer.append_data(frame)
        written += 1
        if (
            live_original is not None
            and live_enhanced is not None
            and (local_index == frame_count - 1 or written == total_frames)
        ):
            save_live_frame_pair(
                live_original,
                live_enhanced,
                original_frames[local_index],
                frame,
                live_quality,
            )
            print(f"LIVE_FRAME {written} {total_frames} {(written - 1) / max(1.0, fps):.6f}", flush=True)
        if written == 1 or written == total_frames or written % 8 == 0:
            print(f"{written - 1} {total_frames}", flush=True)
    return written


def pipeline_module(pipe):
    return importlib.import_module(pipe.__class__.__module__)


def ensure_temporal_rope_capacity(dit, required_positions: int) -> None:
    freqs = getattr(dit, "freqs", None)
    if not freqs or len(freqs) < 3:
        return

    temporal_freqs = freqs[0]
    if temporal_freqs.shape[0] >= required_positions:
        return

    model_module = importlib.import_module(dit.__class__.__module__)
    precompute = getattr(model_module, "precompute_freqs_cis", None)
    if precompute is None:
        raise RuntimeError("Could not extend FlashVSR temporal RoPE frequencies.")

    temporal_dim = int(temporal_freqs.shape[-1]) * 2
    extended = precompute(temporal_dim, end=required_positions).to(device=temporal_freqs.device)
    if extended.dtype != temporal_freqs.dtype:
        extended = extended.to(dtype=temporal_freqs.dtype)

    dit.freqs = (extended, freqs[1], freqs[2])
    print(
        f"Extended FlashVSR temporal RoPE positions: {temporal_freqs.shape[0]} -> {required_positions}",
        flush=True,
    )


def stream_flashvsr_tiny(module, pipe, args) -> None:
    source = StreamingLqFrameBuffer(
        module,
        args.input.resolve(),
        scale=args.scale,
        dtype=torch.bfloat16,
        expected_frames=args.total_frames,
        fps_override=args.fps,
    )
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output_path), fps=source.fps, quality=args.quality)
    device = pipe.device
    height, width = pipe.check_resize_height_width(source.target_height, source.target_width)
    if height != source.target_height or width != source.target_width:
        raise RuntimeError(
            f"FlashVSR streaming expected pre-aligned dimensions, got {source.target_width}x{source.target_height} "
            f"but pipeline requested {width}x{height}."
        )

    if hasattr(pipe.dit, "LQ_proj_in"):
        pipe.dit.LQ_proj_in.clear_cache()
    pipe.TCDecoder.clean_mem()
    pipe_module = pipeline_module(pipe)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    process_total_num = (source.model_frames - 1) // 8 - 2
    if process_total_num <= 0:
        raise RuntimeError(f"FlashVSR streaming needs at least one process step, got {process_total_num}.")
    required_temporal_positions = max(6, 4 + max(0, process_total_num - 1) * 2 + 2)
    ensure_temporal_rope_capacity(pipe.dit, required_temporal_positions)

    pre_cache_k = None
    pre_cache_v = None
    lq_pre_idx = 0
    written = 0

    try:
        with torch.no_grad():
            for cur_process_idx in range(process_total_num):
                if cur_process_idx == 0:
                    pre_cache_k = [None] * len(pipe.dit.blocks)
                    pre_cache_v = [None] * len(pipe.dit.blocks)
                    lq_latents = None
                    inner_loop_num = 7
                    for inner_idx in range(inner_loop_num):
                        start = max(0, inner_idx * 4 - 3)
                        end = (inner_idx + 1) * 4 - 3
                        cur = pipe.denoising_model().LQ_proj_in.stream_forward(
                            source.slice(start, end).to(device)
                        )
                        if cur is None:
                            continue
                        if lq_latents is None:
                            lq_latents = cur
                        else:
                            for layer_idx in range(len(lq_latents)):
                                lq_latents[layer_idx] = torch.cat([lq_latents[layer_idx], cur[layer_idx]], dim=1)
                    lq_cur_idx = (inner_loop_num - 1) * 4 - 3
                    cur_latents = torch.randn(
                        (1, 16, 6, height // 8, width // 8),
                        generator=generator,
                        device=device,
                        dtype=pipe.torch_dtype,
                    )
                else:
                    lq_latents = None
                    inner_loop_num = 2
                    for inner_idx in range(inner_loop_num):
                        start = cur_process_idx * 8 + 17 + inner_idx * 4
                        end = cur_process_idx * 8 + 21 + inner_idx * 4
                        cur = pipe.denoising_model().LQ_proj_in.stream_forward(
                            source.slice(start, end).to(device)
                        )
                        if cur is None:
                            continue
                        if lq_latents is None:
                            lq_latents = cur
                        else:
                            for layer_idx in range(len(lq_latents)):
                                lq_latents[layer_idx] = torch.cat([lq_latents[layer_idx], cur[layer_idx]], dim=1)
                    lq_cur_idx = cur_process_idx * 8 + 21 + (inner_loop_num - 2) * 4
                    cur_latents = torch.randn(
                        (1, 16, 2, height // 8, width // 8),
                        generator=generator,
                        device=device,
                        dtype=pipe.torch_dtype,
                    )

                noise_pred_posi, pre_cache_k, pre_cache_v = pipe_module.model_fn_wan_video(
                    pipe.dit,
                    x=cur_latents,
                    timestep=pipe.timestep,
                    context=None,
                    tea_cache=None,
                    use_unified_sequence_parallel=False,
                    LQ_latents=lq_latents,
                    is_full_block=False,
                    is_stream=True,
                    pre_cache_k=pre_cache_k,
                    pre_cache_v=pre_cache_v,
                    topk_ratio=args.sparse_ratio * 768 * 1280 / (height * width),
                    kv_ratio=3.0,
                    cur_process_idx=cur_process_idx,
                    t_mod=pipe.t_mod,
                    t=pipe.t,
                    local_range=args.local_range,
                )

                cur_latents = cur_latents - noise_pred_posi
                del noise_pred_posi, lq_latents
                # Match the official FlashVSR streaming pipeline: keep the DIT
                # under its per-layer VRAM wrappers instead of full onload/offload
                # cycles between every temporal step.
                torch.cuda.empty_cache()
                cur_lq_frame = source.slice(lq_pre_idx, lq_cur_idx).to(device)
                cur_frames = pipe.TCDecoder.decode_video(
                    cur_latents.transpose(1, 2),
                    parallel=False,
                    show_progress_bar=False,
                    cond=cur_lq_frame,
                ).transpose(1, 2).mul_(2).sub_(1)

                try:
                    cur_frames = pipe.ColorCorrector(
                        cur_frames.to(device=device),
                        cur_lq_frame,
                        clip_range=(-1, 1),
                        chunk_size=None,
                        method="adain",
                    )
                except Exception:
                    pass

                remaining = source.source_frames - written
                if remaining > 0:
                    written = append_video_frames(
                        writer,
                        cur_frames[0],
                        cur_lq_frame[0],
                        remaining,
                        source.source_frames,
                        written,
                        fps=source.fps,
                        live_original=args.live_original,
                        live_enhanced=args.live_enhanced,
                        live_quality=args.live_quality,
                    )
                lq_pre_idx = lq_cur_idx
                source.discard_before(lq_pre_idx)
                del cur_latents, cur_lq_frame, cur_frames
                torch.cuda.empty_cache()

        if written != source.source_frames:
            raise RuntimeError(f"FlashVSR streaming wrote {written} frames, expected {source.source_frames}.")
    finally:
        writer.close()
        source.close()


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
    parser.add_argument("--total_frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--live_original", type=Path, default=None)
    parser.add_argument("--live_enhanced", type=Path, default=None)
    parser.add_argument("--live_quality", type=int, default=92)
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
    if args.streaming:
        if args.variant not in {"tiny_long", "tiny"}:
            raise RuntimeError("FlashVSR streaming is currently supported only for the tiny and tiny_long variants.")
        stream_flashvsr_tiny(module, pipe, args)
        print(f"Done: {output_path}")
        return 0

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
