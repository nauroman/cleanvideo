from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image


TemporalConsistency = Literal["off", "light", "medium", "strong", "extra_strong"]


@dataclass(frozen=True)
class TemporalProfile:
    blend: float
    diff_sigma: float
    scene_cut_threshold: float
    max_analysis_side: int


TEMPORAL_PROFILES: dict[TemporalConsistency, TemporalProfile] = {
    "off": TemporalProfile(blend=0.0, diff_sigma=1.0, scene_cut_threshold=1.0, max_analysis_side=960),
    "light": TemporalProfile(blend=0.16, diff_sigma=18.0, scene_cut_threshold=42.0, max_analysis_side=960),
    "medium": TemporalProfile(blend=0.26, diff_sigma=22.0, scene_cut_threshold=48.0, max_analysis_side=960),
    "strong": TemporalProfile(blend=0.38, diff_sigma=28.0, scene_cut_threshold=56.0, max_analysis_side=960),
    "extra_strong": TemporalProfile(blend=0.54, diff_sigma=38.0, scene_cut_threshold=68.0, max_analysis_side=960),
}


def temporal_mode_enabled(mode: TemporalConsistency) -> bool:
    return TEMPORAL_PROFILES[mode].blend > 0


def stabilize_frame(
    *,
    previous_source_path: Path,
    current_source_path: Path,
    previous_enhanced_path: Path,
    current_enhanced_path: Path,
    output_path: Path,
    mode: TemporalConsistency,
) -> bool:
    if not temporal_mode_enabled(mode):
        if current_enhanced_path != output_path:
            Image.open(current_enhanced_path).save(output_path)
        return False

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Temporal consistency needs opencv-python-headless. Run scripts\\setup.ps1 to install updated dependencies."
        ) from exc

    profile = TEMPORAL_PROFILES[mode]
    current_enhanced = _read_rgb(current_enhanced_path)
    previous_enhanced = _read_rgb(previous_enhanced_path)
    if current_enhanced.shape != previous_enhanced.shape:
        previous_enhanced = cv2.resize(
            previous_enhanced,
            (current_enhanced.shape[1], current_enhanced.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

    previous_source = _read_rgb(previous_source_path)
    current_source = _read_rgb(current_source_path)
    analysis_size = _analysis_size(current_source.shape[1], current_source.shape[0], profile.max_analysis_side)
    previous_analysis = cv2.resize(previous_source, analysis_size, interpolation=cv2.INTER_AREA)
    current_analysis = cv2.resize(current_source, analysis_size, interpolation=cv2.INTER_AREA)

    prev_gray = cv2.cvtColor(previous_analysis, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(current_analysis, cv2.COLOR_RGB2GRAY)
    global_diff = float(np.mean(cv2.absdiff(prev_gray, curr_gray)))
    if global_diff > profile.scene_cut_threshold:
        Image.fromarray(current_enhanced).save(output_path)
        return False

    # Backward flow maps each current-frame pixel to the matching previous-frame pixel.
    backward_flow = cv2.calcOpticalFlowFarneback(
        curr_gray,
        prev_gray,
        None,
        pyr_scale=0.5,
        levels=4,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )
    output_h, output_w = current_enhanced.shape[:2]
    flow = cv2.resize(backward_flow, (output_w, output_h), interpolation=cv2.INTER_LINEAR)
    flow[..., 0] *= output_w / analysis_size[0]
    flow[..., 1] *= output_h / analysis_size[1]

    grid_x, grid_y = np.meshgrid(np.arange(output_w, dtype=np.float32), np.arange(output_h, dtype=np.float32))
    map_x = grid_x + flow[..., 0].astype(np.float32)
    map_y = grid_y + flow[..., 1].astype(np.float32)
    warped_previous = cv2.remap(
        previous_enhanced,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )

    previous_source_output = cv2.resize(previous_source, (output_w, output_h), interpolation=cv2.INTER_CUBIC)
    current_source_output = cv2.resize(current_source, (output_w, output_h), interpolation=cv2.INTER_CUBIC)
    warped_previous_source = cv2.remap(
        previous_source_output,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )

    local_diff = np.mean(
        np.abs(current_source_output.astype(np.float32) - warped_previous_source.astype(np.float32)),
        axis=2,
        keepdims=True,
    )
    confidence = np.exp(-local_diff / profile.diff_sigma).astype(np.float32)
    confidence = cv2.GaussianBlur(confidence, (0, 0), sigmaX=1.1, sigmaY=1.1)
    if confidence.ndim == 2:
        confidence = confidence[..., None]

    weight = np.clip(profile.blend * confidence, 0.0, profile.blend)
    stabilized = (
        current_enhanced.astype(np.float32) * (1.0 - weight)
        + warped_previous.astype(np.float32) * weight
    )
    Image.fromarray(np.clip(stabilized, 0, 255).astype(np.uint8)).save(output_path)
    return True


def _read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _analysis_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    longest = max(width, height)
    if longest <= max_side:
        return width, height
    scale = max_side / longest
    return max(2, int(round(width * scale))), max(2, int(round(height * scale)))
