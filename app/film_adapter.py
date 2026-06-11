from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from PIL import Image


PATCH_SIZE = 512
TRAINING_PACKAGE_VERSIONS = {
    "open_clip_torch": "2.31.0",
    "polars": "1.24.0",
    "timm": "1.0.15",
}
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
class AdapterDataset:
    frames: int
    patches: int
    parquet_path: Path
    config_path: Path
    output_dir: Path


def build_adapter_dataset(
    *,
    video_path: Path,
    duration_seconds: float,
    adapter_root: Path,
    base_model_path: Path,
    prompt: str,
    max_frames: int,
    patches_per_frame: int,
    max_train_steps: int,
) -> AdapterDataset:
    validate_training_dependencies()
    source_frames_dir = adapter_root / "source_frames"
    patches_dir = adapter_root / "patches"
    output_dir = adapter_root / "training"
    for directory in [source_frames_dir, patches_dir, output_dir]:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    extracted = extract_sample_frames(video_path, source_frames_dir, duration_seconds, max_frames)
    patch_paths = write_training_patches(
        frame_paths=extracted,
        patches_dir=patches_dir,
        patches_per_frame=patches_per_frame,
    )
    if not patch_paths:
        raise RuntimeError("Could not create any 512x512 training patches from this video.")

    parquet_path = adapter_root / "dataset.parquet"
    write_parquet(patch_paths, prompt, parquet_path)
    config_path = adapter_root / "hypir_train.yaml"
    write_train_config(
        config_path=config_path,
        output_dir=output_dir,
        parquet_path=parquet_path,
        base_model_path=base_model_path,
        max_train_steps=max_train_steps,
    )
    (adapter_root / "dataset.json").write_text(
        json.dumps(
            {
                "frames": len(extracted),
                "patches": len(patch_paths),
                "prompt": prompt,
                "maxTrainSteps": max_train_steps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return AdapterDataset(
        frames=len(extracted),
        patches=len(patch_paths),
        parquet_path=parquet_path,
        config_path=config_path,
        output_dir=output_dir,
    )


def validate_training_dependencies() -> None:
    mismatches: list[str] = []
    for package, expected_version in TRAINING_PACKAGE_VERSIONS.items():
        try:
            installed_version = metadata.version(package)
        except metadata.PackageNotFoundError:
            mismatches.append(f"{package} is not installed")
            continue
        if installed_version != expected_version:
            mismatches.append(f"{package}=={installed_version}, expected {expected_version}")
    if mismatches:
        raise RuntimeError(
            "Film adapter training dependencies do not match HYPIR. "
            + "; ".join(mismatches)
            + ". Reinstall with: .\\.venv\\Scripts\\python.exe -m pip install -r requirements-inference.txt"
        )


def extract_sample_frames(
    video_path: Path,
    output_dir: Path,
    duration_seconds: float,
    max_frames: int,
) -> list[Path]:
    fps = 1.0
    if duration_seconds > 0:
        fps = max_frames / duration_seconds
    fps = min(2.0, max(0.02, fps))
    args = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-vf",
        f"fps={fps:.6f}",
        "-frames:v",
        str(max_frames),
        str(output_dir / "sample_%06d.png"),
    ]
    subprocess.run(args, text=True, capture_output=True, check=True)
    frames = sorted(output_dir.glob("sample_*.png"))
    if not frames:
        raise RuntimeError("ffmpeg did not extract training sample frames.")
    return frames


def write_training_patches(
    *,
    frame_paths: list[Path],
    patches_dir: Path,
    patches_per_frame: int,
) -> list[Path]:
    patch_paths: list[Path] = []
    for frame_index, frame_path in enumerate(frame_paths, start=1):
        image = Image.open(frame_path).convert("RGB")
        image = normalize_training_frame(image)
        crop_boxes = crop_boxes_for(image.width, image.height, patches_per_frame)
        for crop_index, box in enumerate(crop_boxes, start=1):
            patch = image.crop(box)
            patch_path = patches_dir / f"patch_{frame_index:06d}_{crop_index:02d}.png"
            patch.save(patch_path)
            patch_paths.append(patch_path)
    return patch_paths


def normalize_training_frame(image: Image.Image, max_longest_side: int = 1536) -> Image.Image:
    width, height = image.size
    shortest = min(width, height)
    if shortest < PATCH_SIZE:
        scale = PATCH_SIZE / max(1, shortest)
        image = image.resize(
            (max(PATCH_SIZE, round(width * scale)), max(PATCH_SIZE, round(height * scale))),
            Image.Resampling.BICUBIC,
        )
        width, height = image.size

    longest = max(width, height)
    if longest > max_longest_side:
        scale = max_longest_side / longest
        image = image.resize(
            (max(PATCH_SIZE, round(width * scale)), max(PATCH_SIZE, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
    return image


def crop_boxes_for(width: int, height: int, count: int) -> list[tuple[int, int, int, int]]:
    max_x = max(0, width - PATCH_SIZE)
    max_y = max(0, height - PATCH_SIZE)
    candidates = [
        (max_x // 2, max_y // 2),
        (0, 0),
        (max_x, 0),
        (0, max_y),
        (max_x, max_y),
        (max_x // 4, max_y // 4),
        (max_x * 3 // 4, max_y // 4),
        (max_x // 4, max_y * 3 // 4),
        (max_x * 3 // 4, max_y * 3 // 4),
    ]
    boxes: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for x, y in candidates:
        x = min(max(0, x), max_x)
        y = min(max(0, y), max_y)
        if (x, y) in seen:
            continue
        seen.add((x, y))
        boxes.append((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
        if len(boxes) >= count:
            break
    return boxes


def write_parquet(patch_paths: list[Path], prompt: str, parquet_path: Path) -> None:
    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError("Film adapter training needs polars. Run scripts\\setup.ps1 to install training dependencies.") from exc
    df = pl.DataFrame(
        {
            "image_path": [str(path) for path in patch_paths],
            "prompt": [prompt] * len(patch_paths),
        }
    )
    df.write_parquet(parquet_path)


def write_train_config(
    *,
    config_path: Path,
    output_dir: Path,
    parquet_path: Path,
    base_model_path: Path,
    max_train_steps: int,
) -> None:
    checkpointing_steps = max(50, math.ceil(max_train_steps / 4))
    modules = "[" + ", ".join(LORA_MODULES) + "]"
    config_path.write_text(
        f"""output_dir: {output_dir.as_posix()}

data_config:
  train:
    batch_size: 1
    dataloader_num_workers: 0
    dataset:
      target: HYPIR.dataset.realesrgan.RealESRGANDataset
      params:
        file_meta:
          file_list: {parquet_path.as_posix()}
          image_path_prefix: ""
          image_path_key: image_path
          prompt_key: prompt
        file_backend_cfg:
          target: HYPIR.dataset.file_backend.HardDiskBackend
        out_size: {PATCH_SIZE}
        crop_type: none
        use_hflip: true
        use_rot: false
        blur_kernel_size: 21
        kernel_list: ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        kernel_prob: [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        sinc_prob: 0.1
        blur_sigma: [0.2, 3]
        betag_range: [0.5, 4]
        betap_range: [1, 2]
        blur_kernel_size2: 21
        kernel_list2: ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        kernel_prob2: [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        sinc_prob2: 0.1
        blur_sigma2: [0.2, 1.5]
        betag_range2: [0.5, 4]
        betap_range2: [1, 2]
        final_sinc_prob: 0.8
        p_empty_prompt: 0.0
    batch_transform:
      target: HYPIR.dataset.batch_transform.RealESRGANBatchTransform
      params:
        hq_key: hq
        extra_keys: [txt]
        use_sharpener: true
        queue_size: 16
        resize_prob: [0.2, 0.7, 0.1]
        resize_range: [0.15, 1.5]
        gaussian_noise_prob: 0.5
        noise_range: [1, 30]
        poisson_scale_range: [0.05, 3]
        gray_noise_prob: 0.4
        jpeg_range: [30, 95]
        stage2_scale: 4
        second_blur_prob: 0.8
        resize_prob2: [0.3, 0.4, 0.3]
        resize_range2: [0.3, 1.2]
        gaussian_noise_prob2: 0.5
        noise_range2: [1, 25]
        poisson_scale_range2: [0.05, 2.5]
        gray_noise_prob2: 0.4
        jpeg_range2: [30, 95]
        resize_back: true

base_model_type: sd2
base_model_path: {base_model_path.as_posix()}
model_t: 200
coeff_t: 200
lora_rank: 256
lora_modules: {modules}
use_ema: true
ema_decay: 0.999
resume_ema: true
lambda_gan: 0.5
lambda_lpips: 5
lambda_l2: 1
lr_G: 5e-6
lr_D: 5e-6
optimizer_type: adam
opt_kwargs:
  betas: [0.9, 0.999]
mixed_precision: bf16
seed: 231
max_train_steps: {max_train_steps}
gradient_accumulation_steps: 1
gradient_checkpointing: true
max_grad_norm: 1.0
logging_dir: logs
report_to: ~
checkpointing_steps: {checkpointing_steps}
checkpoints_total_limit: 2
resume_from_checkpoint: ~
log_image_steps: {max_train_steps + 1}
log_grad_steps: {max_train_steps + 1}
log_grad_modules: [conv_out]
""",
        encoding="utf-8",
    )
