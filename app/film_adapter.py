from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np
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
    candidates: int
    frames: int
    patches: int
    max_train_steps: int
    checkpointing_steps: int
    checkpoints_total_limit: int
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
    min_train_steps: int,
    max_train_steps: int,
    train_steps_per_patch: float,
    checkpoints_total_limit: int,
) -> AdapterDataset:
    validate_training_dependencies()
    source_frames_dir = adapter_root / "source_frames"
    patches_dir = adapter_root / "patches"
    output_dir = adapter_root / "training"
    for directory in [source_frames_dir, patches_dir, output_dir]:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    candidate_count = min(max(max_frames * 2, max_frames + 128), max_frames + 2048)
    candidates = extract_sample_frames(video_path, source_frames_dir, duration_seconds, candidate_count)
    extracted = select_training_frames(candidates, max_frames)
    patch_paths = write_training_patches(
        frame_paths=extracted,
        patches_dir=patches_dir,
        patches_per_frame=patches_per_frame,
    )
    if not patch_paths:
        raise RuntimeError("Could not create any 512x512 training patches from this video.")

    max_train_steps = training_step_count(
        patch_count=len(patch_paths),
        min_train_steps=min_train_steps,
        max_train_steps=max_train_steps,
        train_steps_per_patch=train_steps_per_patch,
    )
    checkpointing_steps = checkpoint_interval(max_train_steps, checkpoints_total_limit)
    parquet_path = adapter_root / "dataset.parquet"
    write_parquet(patch_paths, prompt, parquet_path)
    config_path = adapter_root / "hypir_train.yaml"
    write_train_config(
        config_path=config_path,
        output_dir=output_dir,
        parquet_path=parquet_path,
        base_model_path=base_model_path,
        max_train_steps=max_train_steps,
        checkpointing_steps=checkpointing_steps,
        checkpoints_total_limit=checkpoints_total_limit,
    )
    (adapter_root / "dataset.json").write_text(
        json.dumps(
            {
                "candidates": len(candidates),
                "frames": len(extracted),
                "patches": len(patch_paths),
                "prompt": prompt,
                "maxTrainSteps": max_train_steps,
                "checkpointingSteps": checkpointing_steps,
                "checkpointsTotalLimit": checkpoints_total_limit,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return AdapterDataset(
        candidates=len(candidates),
        frames=len(extracted),
        patches=len(patch_paths),
        max_train_steps=max_train_steps,
        checkpointing_steps=checkpointing_steps,
        checkpoints_total_limit=checkpoints_total_limit,
        parquet_path=parquet_path,
        config_path=config_path,
        output_dir=output_dir,
    )


def training_step_count(
    *,
    patch_count: int,
    min_train_steps: int,
    max_train_steps: int,
    train_steps_per_patch: float,
) -> int:
    min_train_steps = max(20, min_train_steps)
    max_train_steps = max(min_train_steps, max_train_steps)
    adaptive_steps = math.ceil(max(1, patch_count) * max(0.0, train_steps_per_patch))
    return min(max_train_steps, max(min_train_steps, adaptive_steps))


def checkpoint_interval(max_train_steps: int, checkpoints_total_limit: int) -> int:
    checkpoints_total_limit = max(1, checkpoints_total_limit)
    return max(25, math.ceil(max_train_steps / checkpoints_total_limit))


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
    fps = min(2.0, max(0.001, fps))
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


@dataclass(frozen=True)
class FrameAnalysis:
    path: Path
    brightness: float
    contrast: float
    sharpness: float
    signature: int

    @property
    def usable(self) -> bool:
        return 16 <= self.brightness <= 240 and self.contrast >= 8 and self.sharpness >= 12


def analyze_frame(frame_path: Path) -> FrameAnalysis:
    image = Image.open(frame_path).convert("L")
    analysis = image.copy()
    analysis.thumbnail((256, 256), Image.Resampling.BICUBIC)
    pixels = np.asarray(analysis, dtype=np.float32)
    brightness = float(pixels.mean())
    contrast = float(pixels.std())
    sharpness = laplacian_variance(pixels)
    signature = average_hash(image)
    return FrameAnalysis(
        path=frame_path,
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        signature=signature,
    )


def laplacian_variance(pixels: np.ndarray) -> float:
    center = pixels[1:-1, 1:-1] * 4
    laplacian = center - pixels[:-2, 1:-1] - pixels[2:, 1:-1] - pixels[1:-1, :-2] - pixels[1:-1, 2:]
    return float(laplacian.var())


def average_hash(image: Image.Image) -> int:
    thumbnail = image.resize((8, 8), Image.Resampling.BICUBIC)
    pixels = np.asarray(thumbnail, dtype=np.float32)
    threshold = float(pixels.mean())
    signature = 0
    for index, value in enumerate(pixels.flatten()):
        if value >= threshold:
            signature |= 1 << index
    return signature


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def select_training_frames(frame_paths: list[Path], max_frames: int) -> list[Path]:
    analyses = [analyze_frame(path) for path in frame_paths]
    usable = [item for item in analyses if item.usable]
    if len(usable) < max(4, max_frames // 4):
        usable = analyses

    unique: list[FrameAnalysis] = []
    for item in usable:
        if all(hamming_distance(item.signature, previous.signature) >= 7 for previous in unique):
            unique.append(item)
    if len(unique) < max(4, max_frames // 3):
        unique = usable

    selected = evenly_sample(unique, max_frames)
    return [item.path for item in selected]


def evenly_sample(items: list[FrameAnalysis], max_items: int) -> list[FrameAnalysis]:
    if len(items) <= max_items:
        return items
    if max_items <= 1:
        return items[:max_items]
    selected: list[FrameAnalysis] = []
    last_index = -1
    for slot in range(max_items):
        index = round(slot * (len(items) - 1) / (max_items - 1))
        if index == last_index:
            index = min(len(items) - 1, index + 1)
        selected.append(items[index])
        last_index = index
    return selected


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
        candidate_boxes = crop_boxes_for(image.width, image.height, max(patches_per_frame * 4, patches_per_frame))
        crop_boxes = select_crop_boxes(image, candidate_boxes, patches_per_frame)
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


def select_crop_boxes(
    image: Image.Image,
    candidate_boxes: list[tuple[int, int, int, int]],
    count: int,
) -> list[tuple[int, int, int, int]]:
    scored = [(score_crop(image, box), box) for box in candidate_boxes]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [box for _, box in scored[:count]]


def score_crop(image: Image.Image, box: tuple[int, int, int, int]) -> float:
    crop = image.crop(box).convert("L")
    crop.thumbnail((160, 160), Image.Resampling.BICUBIC)
    pixels = np.asarray(crop, dtype=np.float32)
    brightness = float(pixels.mean())
    contrast = float(pixels.std())
    sharpness = laplacian_variance(pixels)
    exposure_penalty = abs(brightness - 118) * 0.1
    return contrast + math.sqrt(max(0.0, sharpness)) - exposure_penalty


def crop_boxes_for(width: int, height: int, count: int) -> list[tuple[int, int, int, int]]:
    max_x = max(0, width - PATCH_SIZE)
    max_y = max(0, height - PATCH_SIZE)
    positions = [
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
    grid_steps = max(2, math.ceil(math.sqrt(max(count, 1))) + 1)
    for y_step in range(grid_steps):
        for x_step in range(grid_steps):
            x = round(max_x * x_step / max(1, grid_steps - 1))
            y = round(max_y * y_step / max(1, grid_steps - 1))
            positions.append((x, y))

    boxes: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for x, y in positions:
        x = min(max(0, x), max_x)
        y = min(max(0, y), max_y)
        if (x, y) in seen:
            continue
        seen.add((x, y))
        boxes.append((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
    return boxes[:count]


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
    checkpointing_steps: int,
    checkpoints_total_limit: int,
) -> None:
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
checkpoints_total_limit: {checkpoints_total_limit}
resume_from_checkpoint: ~
log_image_steps: {max_train_steps + 1}
log_grad_steps: {max_train_steps + 1}
log_grad_modules: [conv_out]
""",
        encoding="utf-8",
    )


def set_train_config_resume_checkpoint(config_path: Path, checkpoint_dir: Path | None) -> None:
    resume_value = "~" if checkpoint_dir is None else checkpoint_dir.as_posix()
    replacement = f"resume_from_checkpoint: {resume_value}"
    lines = config_path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.startswith("resume_from_checkpoint:"):
            lines[index] = replacement
            break
    else:
        lines.append(replacement)
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cleanup_adapter_training_artifacts(adapter_root: Path, keep_checkpoint_dir: Path) -> None:
    adapter_root = adapter_root.resolve()
    keep_checkpoint_dir = keep_checkpoint_dir.resolve()
    ensure_child_path(adapter_root, keep_checkpoint_dir)

    for directory_name in ["source_frames", "patches"]:
        remove_path(adapter_root / directory_name)
    for file_name in ["dataset.parquet", "dataset.json", "hypir_train.yaml", "train.log"]:
        remove_path(adapter_root / file_name)

    training_dir = adapter_root / "training"
    if not training_dir.exists():
        return

    remove_path(training_dir / "logs")
    for checkpoint_dir in training_dir.glob("checkpoint-*"):
        if not checkpoint_dir.is_dir():
            continue
        if checkpoint_dir.resolve() == keep_checkpoint_dir:
            compact_adapter_checkpoint(checkpoint_dir)
        else:
            remove_path(checkpoint_dir)


def compact_adapter_checkpoint(checkpoint_dir: Path) -> None:
    state_dict_path = checkpoint_dir / "state_dict.pth"
    if not state_dict_path.exists():
        raise RuntimeError(f"Cannot compact adapter checkpoint without {state_dict_path}")
    for child in checkpoint_dir.iterdir():
        if child.name == "state_dict.pth":
            continue
        remove_path(child)


def ensure_child_path(root: Path, path: Path) -> None:
    root = root.resolve()
    path = path.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Refusing to clean outside adapter root: {path}")


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
