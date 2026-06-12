from __future__ import annotations

import json
import hashlib
import math
import os
import shutil
import stat
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from .film_adapter import (
    build_adapter_dataset,
    cleanup_adapter_training_artifacts,
    set_train_config_resume_checkpoint,
)
from .hypir_engine import HypirSettings, engine
from .temporal_stabilizer import TemporalConsistency, stabilize_frame, temporal_mode_enabled
from .video_ops import (
    encode_video,
    extract_frame,
    extract_frames,
    h264_nvenc_available,
    probe_video,
    safe_name,
)


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "static"
HYPIR_ROOT = ROOT / "external" / "HYPIR"
WORK_DIR = ROOT / "work"
UPLOAD_DIR = WORK_DIR / "uploads"
PREVIEW_DIR = WORK_DIR / "previews"
EXPORT_DIR = WORK_DIR / "exports"
JOB_DIR = WORK_DIR / "jobs"
CACHE_DIR = WORK_DIR / "cache"
PARTIAL_DIR = WORK_DIR / "partials"
ADAPTER_DIR = WORK_DIR / "adapters"
for directory in [UPLOAD_DIR, PREVIEW_DIR, EXPORT_DIR, JOB_DIR, CACHE_DIR, PARTIAL_DIR, ADAPTER_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
GENERATED_DIRS = [PREVIEW_DIR, CACHE_DIR, PARTIAL_DIR, JOB_DIR, EXPORT_DIR]
APP_BUILD = "2026-06-12-adapter-recovery-v1"
ADAPTER_TRAINING_RECOVERY_LIMIT = 3

AdapterQuality = Literal["fast", "high", "extra"]
SecondPassMode = Literal["off", "base_after_adapter"]
ADAPTER_QUALITY_PRESETS: dict[str, dict[str, float]] = {
    "fast": {
        "minFrames": 64,
        "maxFrames": 160,
        "secondsPerFrame": 45,
        "patchesPerFrame": 1,
        "minTrainSteps": 240,
        "maxTrainSteps": 600,
        "trainStepsPerPatch": 1.2,
        "checkpointsTotalLimit": 2,
    },
    "high": {
        "minFrames": 192,
        "maxFrames": 768,
        "secondsPerFrame": 10,
        "patchesPerFrame": 2,
        "minTrainSteps": 900,
        "maxTrainSteps": 3000,
        "trainStepsPerPatch": 1.3,
        "checkpointsTotalLimit": 4,
    },
    "extra": {
        "minFrames": 480,
        "maxFrames": 1600,
        "secondsPerFrame": 4.5,
        "patchesPerFrame": 2,
        "minTrainSteps": 1500,
        "maxTrainSteps": 6000,
        "trainStepsPerPatch": 1.75,
        "checkpointsTotalLimit": 5,
    },
}


class ProcessSettings(BaseModel):
    engine: Literal["hypir"] = "hypir"
    prompt: str = ""
    scaleBy: Literal["factor", "longest_side"] = "factor"
    upscale: float = Field(default=1, ge=0.5, le=8)
    targetLongestSide: int | None = Field(default=None, ge=256, le=8192)
    patchSize: int = Field(default=512, ge=512, le=1024)
    stride: int = Field(default=256, ge=128, le=1024)
    seed: int = Field(default=231, ge=-1)
    temporalConsistency: TemporalConsistency = "medium"
    adapterId: str = "base"
    secondPass: SecondPassMode = "off"
    device: Literal["cuda"] = "cuda"

    def to_hypir(self, adapter_weight_path: Path | None = None) -> HypirSettings:
        return HypirSettings(
            prompt=self.prompt.strip(),
            scale_by=self.scaleBy,
            upscale=self.upscale,
            target_longest_side=self.targetLongestSide,
            patch_size=self.patchSize,
            stride=self.stride,
            seed=self.seed,
            device=self.device,
            weight_path=str(adapter_weight_path) if adapter_weight_path else None,
        )

    def to_base_refine_hypir(self) -> HypirSettings:
        return HypirSettings(
            prompt=self.prompt.strip(),
            scale_by="factor",
            upscale=1,
            target_longest_side=None,
            patch_size=self.patchSize,
            stride=self.stride,
            seed=self.seed,
            device=self.device,
            weight_path=None,
        )


class PreviewRequest(ProcessSettings):
    videoId: str
    seconds: float = Field(default=0.0, ge=0)


class ExportRequest(ProcessSettings):
    videoId: str
    crf: int = Field(default=18, ge=12, le=32)
    encoder: Literal["auto", "h264_nvenc", "libx264"] = "auto"


class PartialExportRequest(BaseModel):
    crf: int = Field(default=18, ge=12, le=32)
    encoder: Literal["auto", "h264_nvenc", "libx264"] = "auto"


class AdapterTrainRequest(BaseModel):
    videoId: str
    prompt: str = "film-specific restoration, natural detail, consistent texture"
    quality: AdapterQuality = "fast"
    maxFrames: int | None = Field(default=None, ge=4, le=2000)
    patchesPerFrame: int | None = Field(default=None, ge=1, le=9)
    maxTrainSteps: int | None = Field(default=None, ge=20, le=10000)


class JobState(BaseModel):
    id: str
    kind: str
    status: Literal["queued", "running", "done", "error", "cancelled"]
    progress: float
    message: str
    videoId: str | None = None
    etaSeconds: float | None = None
    framesDone: int = 0
    framesTotal: int = 0
    partialFramesReady: int = 0
    currentFrameIndex: int | None = None
    currentFrameSeconds: float | None = None
    currentOriginalUrl: str | None = None
    currentEnhancedUrl: str | None = None
    latestFrameSeq: int = 0
    cacheKey: str | None = None
    cacheHits: int = 0
    cacheMisses: int = 0
    outputUrl: str | None = None
    outputPath: str | None = None
    adapterId: str | None = None
    adapterName: str | None = None
    error: str | None = None
    startedAt: str
    updatedAt: str


class FrameEvent(BaseModel):
    seq: int
    frameIndex: int
    framesTotal: int
    seconds: float
    originalUrl: str
    enhancedUrl: str
    cached: bool = False
    updatedAt: str


app = FastAPI(title="CleanVideo", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8765", "http://localhost:8765"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

videos: dict[str, dict] = {}
adapters: dict[str, dict] = {}
jobs: dict[str, JobState] = {}
job_frame_events: dict[str, list[FrameEvent]] = {}
jobs_lock = threading.Lock()
cancelled_jobs: set[str] = set()
partial_exports_active: set[str] = set()
job_processes: dict[str, subprocess.Popen] = {}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def video_meta_path(video_id: str) -> Path:
    return UPLOAD_DIR / f"{video_id}.json"


def save_video_record(record: dict) -> None:
    video_meta_path(record["id"]).write_text(json.dumps(record, indent=2), encoding="utf-8")


def load_video_records() -> None:
    for meta_path in UPLOAD_DIR.glob("*.json"):
        try:
            record = json.loads(meta_path.read_text(encoding="utf-8"))
            if Path(record["path"]).exists():
                videos[record["id"]] = record
        except Exception:
            continue


def get_video(video_id: str) -> dict:
    record = videos.get(video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found")
    return record


def adapter_meta_path(adapter_id: str) -> Path:
    return ADAPTER_DIR / adapter_id / "adapter.json"


def root_adapter_id(adapter_id: str) -> str:
    return adapter_id.split("@step-", 1)[0]


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def ensure_adapter_path(path: Path) -> None:
    root = ADAPTER_DIR.resolve()
    resolved = path.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"Refusing to delete outside adapter directory: {resolved}")


def save_adapter_record(record: dict) -> None:
    path = adapter_meta_path(record["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def load_adapter_records() -> None:
    adapters.clear()
    for meta_path in ADAPTER_DIR.glob("*/adapter.json"):
        try:
            record = json.loads(meta_path.read_text(encoding="utf-8"))
            weight_path = Path(record["weightPath"])
            if weight_path.exists():
                if not record.get("checkpoints"):
                    output_dir = Path(record.get("rootPath") or meta_path.parent) / "training"
                    checkpoints = checkpoint_weights(output_dir)
                    if checkpoints:
                        record["checkpointStep"] = checkpoints[-1]["step"]
                        record["checkpoints"] = [
                            {"step": item["step"], "weightPath": str(item["weightPath"])}
                            for item in checkpoints
                        ]
                record, changed = prune_adapter_record_checkpoints(record)
                if changed:
                    save_adapter_record(record)
                adapters[record["id"]] = record
        except Exception:
            continue


def list_adapter_records() -> list[dict]:
    base = {
        "id": "base",
        "name": "Base HYPIR",
        "status": "ready",
        "weightPath": str(engine.weight_path),
        "createdAt": None,
        "videoId": None,
    }
    ordered = sorted(adapters.values(), key=lambda record: record.get("createdAt") or "", reverse=True)
    result = [base]
    for record in ordered:
        result.extend(adapter_select_records(record))
    return result


def adapter_select_records(record: dict) -> list[dict]:
    checkpoints = record.get("checkpoints") or []
    visible = dict(record)
    if checkpoints:
        latest_step = record.get("checkpointStep") or checkpoints[-1].get("step")
        visible["name"] = f"{record['name']} (latest step {latest_step})"
    return [visible]


def get_adapter_weight_path(adapter_id: str) -> Path | None:
    if adapter_id == "base":
        return None
    checkpoint_step: int | None = None
    root_id = root_adapter_id(adapter_id)
    if "@step-" in adapter_id:
        _, step_text = adapter_id.split("@step-", 1)
        try:
            checkpoint_step = int(step_text)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Film adapter checkpoint not found") from exc
    record = adapters.get(adapter_id)
    if not record and checkpoint_step is not None:
        record = adapters.get(root_id)
    if not record:
        raise HTTPException(status_code=404, detail="Film adapter not found")
    if checkpoint_step is None:
        weight_path = Path(record["weightPath"])
    else:
        checkpoint = next(
            (
                item
                for item in record.get("checkpoints", [])
                if item.get("step") == checkpoint_step
            ),
            None,
        )
        if not checkpoint:
            raise HTTPException(status_code=404, detail="Film adapter checkpoint not found")
        weight_path = Path(checkpoint["weightPath"])
    if not weight_path.exists():
        raise HTTPException(status_code=404, detail="Film adapter weights are missing")
    return weight_path


def adapter_root_path(record: dict, adapter_id: str) -> Path:
    root_path = resolve_repo_path(Path(record.get("rootPath") or ADAPTER_DIR / adapter_id))
    ensure_adapter_path(root_path)
    return root_path


def active_job_ids() -> list[str]:
    with jobs_lock:
        return [job.id for job in jobs.values() if job.status in {"queued", "running"}]


def duration_adaptive_frame_count(duration_seconds: float, preset: dict[str, float]) -> int:
    min_frames = int(preset["minFrames"])
    max_frames = int(preset["maxFrames"])
    seconds_per_frame = float(preset["secondsPerFrame"])
    if duration_seconds <= 0 or seconds_per_frame <= 0:
        return min_frames
    duration_target = math.ceil(duration_seconds / seconds_per_frame)
    return min(max_frames, max(min_frames, duration_target))


def adapter_training_params(request: AdapterTrainRequest, duration_seconds: float) -> dict:
    preset = ADAPTER_QUALITY_PRESETS[request.quality]
    params = {
        "maxFrames": duration_adaptive_frame_count(duration_seconds, preset),
        "patchesPerFrame": int(preset["patchesPerFrame"]),
        "minTrainSteps": int(preset["minTrainSteps"]),
        "maxTrainSteps": int(preset["maxTrainSteps"]),
        "trainStepsPerPatch": float(preset["trainStepsPerPatch"]),
        "checkpointsTotalLimit": int(preset["checkpointsTotalLimit"]),
    }
    if request.maxFrames is not None:
        params["maxFrames"] = request.maxFrames
    if request.patchesPerFrame is not None:
        params["patchesPerFrame"] = request.patchesPerFrame
    if request.maxTrainSteps is not None:
        params["minTrainSteps"] = request.maxTrainSteps
        params["maxTrainSteps"] = request.maxTrainSteps
    return params


def update_job(job_id: str, **changes) -> None:
    with jobs_lock:
        job = jobs[job_id]
        data = job.model_dump()
        data.update(changes)
        data["updatedAt"] = now_iso()
        jobs[job_id] = JobState(**data)


def add_frame_event(
    job_id: str,
    frame_index: int,
    frames_total: int,
    seconds: float,
    original_path: Path,
    enhanced_path: Path,
    *,
    cached: bool,
) -> FrameEvent:
    with jobs_lock:
        events = job_frame_events.setdefault(job_id, [])
        event = FrameEvent(
            seq=len(events) + 1,
            frameIndex=frame_index,
            framesTotal=frames_total,
            seconds=seconds,
            originalUrl=media_url(original_path),
            enhancedUrl=media_url(enhanced_path),
            cached=cached,
            updatedAt=now_iso(),
        )
        events.append(event)
        if job_id in jobs:
            data = jobs[job_id].model_dump()
            data["latestFrameSeq"] = event.seq
            data["updatedAt"] = event.updatedAt
            jobs[job_id] = JobState(**data)
        return event


def media_url(path: Path) -> str:
    return f"/media/{path.relative_to(WORK_DIR).as_posix()}"


def valid_image(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def video_fingerprint(record: dict) -> dict:
    path = Path(record["path"])
    stat = path.stat()
    return {
        "id": record["id"],
        "name": record["name"],
        "size": stat.st_size,
        "mtimeNs": stat.st_mtime_ns,
        "metadata": record["metadata"],
    }


def enhancement_settings(request: ExportRequest) -> dict:
    settings = request.model_dump(exclude={"crf", "encoder"}, mode="json")
    upscale = settings.get("upscale")
    if isinstance(upscale, float) and upscale.is_integer():
        settings["upscale"] = int(upscale)
    return settings


def enhancement_cache_key(record: dict, request: ExportRequest) -> str:
    payload = {
        "engineVersion": "hypir-sd2-v1",
        "video": video_fingerprint(record),
        "settings": enhancement_settings(request),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def export_output_path(record: dict, cache_key: str, request: ExportRequest) -> Path:
    payload = {"cacheKey": cache_key, "crf": request.crf, "encoder": request.encoder}
    suffix = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    output_name = f"{Path(record['name']).stem}_hypir_h264_{cache_key[:8]}_q{request.crf}_{suffix}.mp4"
    return EXPORT_DIR / safe_name(output_name)


def partial_output_path(record: dict, cache_key: str, frame_count: int, request: PartialExportRequest) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    stem = safe_name(Path(record["name"]).stem)[:64]
    output_name = (
        f"{stem}_partial_{frame_count:06d}f_{cache_key[:8]}_"
        f"q{request.crf}_{stamp}_{suffix}.mp4"
    )
    return EXPORT_DIR / safe_name(output_name)


def write_cache_manifest(cache_root: Path, record: dict, request: ExportRequest, status: str) -> None:
    manifest = {
        "status": status,
        "updatedAt": now_iso(),
        "video": video_fingerprint(record),
        "settings": enhancement_settings(request),
    }
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def reusable_frames(video_path: Path, frames_dir: Path, expected_count: int) -> list[Path]:
    frames = sorted(frames_dir.glob("frame_*.png"))
    if frames and (expected_count <= 0 or len(frames) == expected_count):
        return frames
    return extract_frames(video_path, frames_dir)


def contiguous_ready_frames(frames_dir: Path) -> list[Path]:
    frames: list[Path] = []
    index = 1
    while True:
        frame_path = frames_dir / f"frame_{index:06d}.png"
        if not valid_image(frame_path):
            break
        frames.append(frame_path)
        index += 1
    return frames


def snapshot_partial_frames(frames: list[Path], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for index, frame_path in enumerate(frames, start=1):
        target_path = target_dir / f"frame_{index:06d}.png"
        try:
            os.link(frame_path, target_path)
        except OSError:
            shutil.copy2(frame_path, target_path)
        if not valid_image(target_path):
            raise RuntimeError(f"Frame {frame_path.name} was not ready for partial export")


def eta_seconds(generation_elapsed: float, completed_misses: int, remaining_misses: int) -> float | None:
    if remaining_misses <= 0:
        return 0.0
    if completed_misses <= 0:
        return None
    elapsed = max(0.0, generation_elapsed)
    return elapsed / completed_misses * remaining_misses


def open_local_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
        return
    subprocess.Popen(["xdg-open", str(path)])


def ensure_work_path(path: Path) -> None:
    root = WORK_DIR.resolve()
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Refusing to clean outside work directory: {resolved}")


def path_usage(path: Path) -> dict:
    if not path.exists():
        return {"files": 0, "dirs": 0, "bytes": 0}
    if path.is_file():
        return {"files": 1, "dirs": 0, "bytes": path.stat().st_size}
    files = 0
    dirs = 0
    bytes_used = 0
    for child in path.rglob("*"):
        if child.is_file():
            files += 1
            bytes_used += child.stat().st_size
        elif child.is_dir():
            dirs += 1
    return {"files": files, "dirs": dirs, "bytes": bytes_used}


def handle_remove_readonly(func, failed_path, _exc_info) -> None:
    try:
        os.chmod(failed_path, stat.S_IWRITE)
        func(failed_path)
    except FileNotFoundError:
        pass


def remove_generated_path(path: Path, *, recreate: bool) -> dict:
    ensure_work_path(path)
    usage = path_usage(path)
    try:
        if path.exists():
            if path.is_file():
                os.chmod(path, stat.S_IWRITE)
                path.unlink()
            else:
                shutil.rmtree(path, onerror=handle_remove_readonly)
    except FileNotFoundError:
        pass
    if recreate:
        path.mkdir(parents=True, exist_ok=True)
    return usage


def schedule_generated_path_cleanup(path: Path, delay_seconds: int = 120) -> dict:
    ensure_work_path(path)
    usage = path_usage(path)

    def cleanup_later() -> None:
        time.sleep(delay_seconds)
        try:
            remove_generated_path(path, recreate=False)
        except Exception:
            pass

    threading.Thread(target=cleanup_later, daemon=True).start()
    return usage


def clear_generated_dirs() -> dict:
    total_files = 0
    total_dirs = 0
    total_bytes = 0
    cleaned: dict[str, dict] = {}
    for directory in GENERATED_DIRS:
        usage = remove_generated_path(directory, recreate=True)
        cleaned[directory.name] = usage
        total_files += usage["files"]
        total_dirs += usage["dirs"]
        total_bytes += usage["bytes"]
    remaining = {
        directory.name: usage
        for directory in GENERATED_DIRS
        if (usage := path_usage(directory))["files"] or usage["dirs"]
    }
    if remaining:
        details = ", ".join(
            f"{name}: {usage['dirs']} dirs, {usage['files']} files"
            for name, usage in remaining.items()
        )
        raise RuntimeError(f"Cleanup incomplete. Remaining generated data: {details}")
    return {
        "cleaned": cleaned,
        "filesDeleted": total_files,
        "directoriesDeleted": total_dirs,
        "bytesFreed": total_bytes,
        "verifiedEmpty": True,
        "uploadsPreserved": True,
    }


@app.on_event("startup")
def startup() -> None:
    load_video_records()
    load_adapter_records()


@app.get("/api/status")
def status() -> dict:
    with jobs_lock:
        active_jobs = sum(1 for job in jobs.values() if job.status in {"queued", "running"})
    return {
        "app": "CleanVideo",
        "build": APP_BUILD,
        "hypir": engine.status(),
        "ffmpeg": True,
        "nvenc": h264_nvenc_available(),
        "videos": len(videos),
        "jobs": len(jobs),
        "activeJobs": active_jobs,
    }


@app.get("/api/videos")
def list_videos() -> dict:
    return {"videos": list(videos.values())}


@app.get("/api/adapters")
def list_adapters() -> dict:
    return {"adapters": list_adapter_records()}


@app.delete("/api/adapters")
def delete_all_adapters() -> dict:
    active_jobs = active_job_ids()
    if active_jobs:
        raise HTTPException(status_code=409, detail="Cannot delete film adapters while a job is running")

    unloaded = engine.unload_if_weight_under(ADAPTER_DIR)
    deleted: list[dict] = []
    total_files = 0
    total_dirs = 0
    total_bytes = 0
    for child in sorted(ADAPTER_DIR.iterdir()):
        if not child.exists():
            continue
        ensure_adapter_path(child)
        usage = remove_generated_path(child, recreate=False)
        deleted.append(
            {
                "id": child.name,
                "filesDeleted": usage["files"],
                "directoriesDeleted": usage["dirs"],
                "bytesFreed": usage["bytes"],
            }
        )
        total_files += usage["files"]
        total_dirs += usage["dirs"]
        total_bytes += usage["bytes"]
    adapters.clear()
    return {
        "deletedAdapters": deleted,
        "adaptersDeleted": len(deleted),
        "filesDeleted": total_files,
        "directoriesDeleted": total_dirs,
        "bytesFreed": total_bytes,
        "unloadedModel": unloaded,
        "basePreserved": True,
    }


@app.delete("/api/adapters/{adapter_id}")
def delete_adapter(adapter_id: str) -> dict:
    root_id = root_adapter_id(adapter_id)
    if root_id == "base":
        raise HTTPException(status_code=400, detail="Base HYPIR cannot be deleted")
    active_jobs = active_job_ids()
    if active_jobs:
        raise HTTPException(status_code=409, detail="Cannot delete film adapters while a job is running")

    if root_id not in adapters:
        load_adapter_records()
    record = adapters.get(root_id)
    if not record:
        raise HTTPException(status_code=404, detail="Film adapter not found")

    root_path = adapter_root_path(record, root_id)
    unloaded = engine.unload_if_weight_under(root_path)
    try:
        usage = remove_generated_path(root_path, recreate=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Film adapter delete failed: {exc}") from exc
    adapters.pop(root_id, None)
    return {
        "deletedAdapterId": root_id,
        "deletedName": record.get("name") or root_id,
        "filesDeleted": usage["files"],
        "directoriesDeleted": usage["dirs"],
        "bytesFreed": usage["bytes"],
        "unloadedModel": unloaded,
    }


def checkpoint_weights(output_dir: Path) -> list[dict]:
    checkpoints = []
    for checkpoint in output_dir.glob("checkpoint-*"):
        if not checkpoint.is_dir():
            continue
        try:
            step = int(checkpoint.name.split("-", 1)[1])
        except Exception:
            continue
        weight_path = checkpoint / "state_dict.pth"
        if weight_path.exists():
            checkpoints.append({"step": step, "weightPath": weight_path})
    return sorted(checkpoints, key=lambda item: item["step"])


def prune_adapter_checkpoints(checkpoints: list[dict], keep_step: int) -> list[dict]:
    kept: list[dict] = []
    for item in checkpoints:
        step = item.get("step")
        weight_path = Path(item.get("weightPath", ""))
        if step == keep_step:
            if weight_path.exists():
                kept.append({"step": step, "weightPath": weight_path})
            continue

        checkpoint_dir = weight_path.parent
        if checkpoint_dir.name.startswith("checkpoint-"):
            ensure_adapter_path(checkpoint_dir)
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir, onerror=handle_remove_readonly)
    return sorted(kept, key=lambda item: item["step"])


def prune_adapter_record_checkpoints(record: dict) -> tuple[dict, bool]:
    checkpoints = []
    for item in record.get("checkpoints") or []:
        try:
            step = int(item["step"])
        except Exception:
            continue
        weight_path = Path(item.get("weightPath", ""))
        if weight_path.exists():
            checkpoints.append({"step": step, "weightPath": weight_path})

    if not checkpoints:
        return record, False

    checkpoints = sorted(checkpoints, key=lambda item: item["step"])
    keep_step = record.get("checkpointStep") or checkpoints[-1]["step"]
    if not any(item["step"] == keep_step for item in checkpoints):
        keep_step = checkpoints[-1]["step"]

    kept = prune_adapter_checkpoints(checkpoints, int(keep_step))
    if not kept:
        return record, False

    new_checkpoints = [
        {"step": item["step"], "weightPath": str(item["weightPath"])}
        for item in kept
    ]
    changed = record.get("checkpointStep") != kept[-1]["step"]
    changed = changed or record.get("weightPath") != str(kept[-1]["weightPath"])
    changed = changed or record.get("checkpoints") != new_checkpoints
    record["checkpointStep"] = kept[-1]["step"]
    record["weightPath"] = str(kept[-1]["weightPath"])
    record["checkpoints"] = new_checkpoints
    return record, changed


def build_adapter_record(
    *,
    adapter_id: str,
    source_record: dict,
    request: AdapterTrainRequest,
    adapter_root: Path,
    prompt: str,
    dataset: object,
    train_params: dict[str, int],
    checkpoints: list[dict],
    recovered: bool,
    partial: bool = False,
    return_code: int | None = None,
) -> dict:
    checkpoint = checkpoints[-1]
    step = checkpoint["step"]
    suffix = f" {request.quality.title()}"
    if partial:
        suffix += f" Partial Step {step}"
    elif recovered:
        suffix += " Recovered"
    adapter_name = f"{Path(source_record['name']).stem} Film Adapter{suffix}"
    return {
        "id": adapter_id,
        "name": adapter_name,
        "status": "partial" if partial else "ready",
        "videoId": request.videoId,
        "videoName": source_record["name"],
        "weightPath": str(checkpoint["weightPath"]),
        "rootPath": str(adapter_root),
        "prompt": prompt,
        "quality": request.quality,
        "candidateFrames": dataset.candidates,
        "frames": dataset.frames,
        "patches": dataset.patches,
        "maxFrames": train_params["maxFrames"],
        "patchesPerFrame": train_params["patchesPerFrame"],
        "maxTrainSteps": train_params["maxTrainSteps"],
        "checkpointingSteps": train_params["checkpointingSteps"],
        "checkpointsTotalLimit": train_params["checkpointsTotalLimit"],
        "checkpointStep": step,
        "recoveredFromCrash": recovered,
        "trainerReturnCode": return_code,
        "checkpoints": [
            {"step": item["step"], "weightPath": str(item["weightPath"])}
            for item in checkpoints
        ],
        "createdAt": now_iso(),
    }


def run_adapter_training(job_id: str, request: AdapterTrainRequest) -> None:
    process: subprocess.Popen | None = None
    try:
        record = get_video(request.videoId)
        adapter_id = job_id
        adapter_root = ADAPTER_DIR / adapter_id
        adapter_root.mkdir(parents=True, exist_ok=True)
        source = Path(record["path"])
        metadata = record["metadata"]
        prompt = request.prompt.strip() or "film-specific restoration, natural detail, consistent texture"
        duration_seconds = float(metadata.get("duration") or 0)
        train_params = adapter_training_params(request, duration_seconds)
        update_job(
            job_id,
            status="running",
            progress=0.04,
            message=f"Sampling film frames for {request.quality} adapter training",
            videoId=request.videoId,
        )
        dataset = build_adapter_dataset(
            video_path=source,
            duration_seconds=duration_seconds,
            adapter_root=adapter_root,
            base_model_path=engine.base_model_path,
            prompt=prompt,
            max_frames=train_params["maxFrames"],
            patches_per_frame=train_params["patchesPerFrame"],
            min_train_steps=train_params["minTrainSteps"],
            max_train_steps=train_params["maxTrainSteps"],
            train_steps_per_patch=train_params["trainStepsPerPatch"],
            checkpoints_total_limit=train_params["checkpointsTotalLimit"],
        )
        train_params["maxTrainSteps"] = dataset.max_train_steps
        train_params["checkpointingSteps"] = dataset.checkpointing_steps
        train_params["checkpointsTotalLimit"] = dataset.checkpoints_total_limit
        if job_id in cancelled_jobs:
            update_job(job_id, status="cancelled", progress=0.12, message="Film adapter training cancelled")
            cancelled_jobs.discard(job_id)
            return

        update_job(
            job_id,
            progress=0.16,
            message=(
                f"Training {request.quality} film adapter on "
                f"{dataset.patches} patches from {dataset.frames} selected frames"
            ),
            framesDone=dataset.patches,
            framesTotal=dataset.patches,
        )
        log_path = adapter_root / "train.log"
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["TOKENIZERS_PARALLELISM"] = "false"
        command = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--num_processes",
            "1",
            str(HYPIR_ROOT / "train.py"),
            "--config",
            str(dataset.config_path),
        ]
        resume_checkpoint_dir: Path | None = None
        resume_step = 0
        recovery_attempts = 0
        recovered_from_crash = False
        completed_from_final_checkpoint = False
        last_trainer_return_code: int | None = None
        with log_path.open("w", encoding="utf-8", errors="replace") as log_fp:
            while True:
                set_train_config_resume_checkpoint(dataset.config_path, resume_checkpoint_dir)
                attempt_label = (
                    f"resume from {resume_checkpoint_dir.name}"
                    if resume_checkpoint_dir is not None
                    else "fresh start"
                )
                log_fp.write(f"\n[{now_iso()}] Starting trainer attempt: {attempt_label}\n")
                log_fp.write(" ".join(command) + "\n")
                log_fp.flush()
                process = subprocess.Popen(
                    command,
                    cwd=HYPIR_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                )
                job_processes[job_id] = process
                last_update = time.monotonic()
                assert process.stdout is not None
                for line in process.stdout:
                    log_fp.write(line)
                    if time.monotonic() - last_update > 5:
                        checkpoints_now = checkpoint_weights(dataset.output_dir)
                        latest_step = checkpoints_now[-1]["step"] if checkpoints_now else 0
                        progress = 0.24
                        if latest_step:
                            progress = min(
                                0.88,
                                0.20 + latest_step / max(train_params["maxTrainSteps"], 1) * 0.68,
                            )
                        update_job(
                            job_id,
                            progress=progress,
                            message=f"Training film adapter: {line.strip()[:140] or 'running'}",
                        )
                        last_update = time.monotonic()
                    if job_id in cancelled_jobs:
                        process.terminate()
                        try:
                            process.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        update_job(job_id, status="cancelled", progress=0.2, message="Film adapter training cancelled")
                        cancelled_jobs.discard(job_id)
                        return
                return_code = process.wait()
                job_processes.pop(job_id, None)
                if return_code == 0:
                    break

                last_trainer_return_code = return_code
                checkpoints = checkpoint_weights(dataset.output_dir)
                latest_checkpoint = checkpoints[-1] if checkpoints else None
                if latest_checkpoint and latest_checkpoint["step"] >= train_params["maxTrainSteps"]:
                    recovered_from_crash = True
                    completed_from_final_checkpoint = True
                    log_fp.write(
                        f"[{now_iso()}] Trainer exited {return_code}, "
                        f"but final checkpoint {latest_checkpoint['step']} is complete.\n"
                    )
                    break

                if not latest_checkpoint:
                    raise RuntimeError(f"HYPIR adapter training failed with exit code {return_code}. See {log_path}")

                latest_step = int(latest_checkpoint["step"])
                if recovery_attempts >= ADAPTER_TRAINING_RECOVERY_LIMIT:
                    progress_note = (
                        f"and did not advance beyond checkpoint {latest_step}"
                        if latest_step <= resume_step
                        else f"after reaching checkpoint {latest_step}"
                    )
                    raise RuntimeError(
                        f"HYPIR adapter training failed with exit code {return_code} "
                        f"after {recovery_attempts} recovery attempts {progress_note}. See {log_path}"
                    )

                recovery_attempts += 1
                recovered_from_crash = True
                if latest_step > resume_step or resume_checkpoint_dir is None:
                    resume_step = latest_step
                    resume_checkpoint_dir = Path(latest_checkpoint["weightPath"]).parent
                log_fp.write(
                    f"[{now_iso()}] Trainer exited {return_code}; "
                    f"resuming from checkpoint {resume_step} "
                    f"({recovery_attempts}/{ADAPTER_TRAINING_RECOVERY_LIMIT}).\n"
                )
                log_fp.flush()
                update_job(
                    job_id,
                    progress=min(0.9, 0.20 + resume_step / max(train_params["maxTrainSteps"], 1) * 0.68),
                    message=(
                        f"Trainer exited with code {return_code}; "
                        f"resuming film adapter from checkpoint {resume_step} "
                        f"({recovery_attempts}/{ADAPTER_TRAINING_RECOVERY_LIMIT})"
                    ),
                    etaSeconds=None,
                )

        checkpoints = checkpoint_weights(dataset.output_dir)
        if not checkpoints:
            raise RuntimeError(f"Training finished, but no checkpoint state_dict.pth was found under {dataset.output_dir}")
        checkpoints = prune_adapter_checkpoints(checkpoints, checkpoints[-1]["step"])
        adapter_record = build_adapter_record(
            adapter_id=adapter_id,
            source_record=record,
            request=request,
            adapter_root=adapter_root,
            prompt=prompt,
            dataset=dataset,
            train_params=train_params,
            checkpoints=checkpoints,
            recovered=recovered_from_crash,
            return_code=last_trainer_return_code,
        )
        adapters[adapter_id] = adapter_record
        cleanup_message = "Cleaned temporary training files"
        try:
            cleanup_adapter_training_artifacts(adapter_root, Path(adapter_record["weightPath"]).parent)
        except Exception as exc:
            cleanup_message = f"Temporary training cleanup failed: {exc}"
        save_adapter_record(adapter_record)
        if completed_from_final_checkpoint:
            ready_message = (
                f"Film adapter ready from final checkpoint {adapter_record['checkpointStep']} "
                f"after trainer exit code {last_trainer_return_code}"
            )
        elif recovered_from_crash:
            ready_message = f"Film adapter ready after recovery: {adapter_record['name']}"
        else:
            ready_message = f"Film adapter ready: {adapter_record['name']}"
        update_job(
            job_id,
            status="done",
            progress=1.0,
            message=f"{ready_message}. {cleanup_message}.",
            adapterId=adapter_id,
            adapterName=adapter_record["name"],
            outputPath=adapter_record["weightPath"],
            etaSeconds=0,
        )
    except Exception as exc:
        update_job(
            job_id,
            status="error",
            progress=0,
            message="Film adapter training failed",
            etaSeconds=None,
            error=f"{exc}\n{traceback.format_exc()}",
        )
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
        job_processes.pop(job_id, None)
        cancelled_jobs.discard(job_id)


@app.post("/api/adapters/train")
def train_adapter(request: AdapterTrainRequest, background_tasks: BackgroundTasks) -> dict:
    get_video(request.videoId)
    with jobs_lock:
        active_jobs = [job.id for job in jobs.values() if job.status in {"queued", "running"}]
    if active_jobs:
        raise HTTPException(status_code=409, detail="Wait for the current job to finish before training a film adapter")
    job_id = uuid.uuid4().hex
    stamp = now_iso()
    with jobs_lock:
        jobs[job_id] = JobState(
            id=job_id,
            kind="adapter",
            status="queued",
            progress=0,
            message="Queued film adapter training",
            videoId=request.videoId,
            startedAt=stamp,
            updatedAt=stamp,
        )
        job_frame_events[job_id] = []
    background_tasks.add_task(run_adapter_training, job_id, request)
    return jobs[job_id].model_dump()


@app.post("/api/open-output-folder")
def open_output_folder() -> dict:
    try:
        open_local_folder(EXPORT_DIR)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not open output folder: {exc}") from exc
    return {"path": str(EXPORT_DIR)}


@app.post("/api/cleanup-generated")
def cleanup_generated() -> dict:
    with jobs_lock:
        active_jobs = [job.id for job in jobs.values() if job.status in {"queued", "running"}]
        active_partials = len(partial_exports_active)
    if active_jobs:
        raise HTTPException(status_code=409, detail="Cannot clean generated files while an export is running")
    if active_partials:
        raise HTTPException(status_code=409, detail="Cannot clean generated files while a partial video is being saved")
    try:
        result = clear_generated_dirs()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {exc}") from exc
    with jobs_lock:
        jobs.clear()
        job_frame_events.clear()
    cancelled_jobs.clear()
    partial_exports_active.clear()
    return result


@app.post("/api/videos")
async def upload_video(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    video_id = uuid.uuid4().hex
    filename = f"{video_id}_{safe_name(file.filename)}"
    target = UPLOAD_DIR / filename
    with target.open("wb") as fp:
        shutil.copyfileobj(file.file, fp)
    try:
        metadata = probe_video(target)
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not read video: {exc}") from exc

    record = {
        "id": video_id,
        "name": file.filename,
        "path": str(target),
        "url": f"/media/uploads/{filename}",
        "metadata": metadata,
        "createdAt": now_iso(),
    }
    videos[video_id] = record
    save_video_record(record)
    return record


@app.post("/api/preview")
def preview_frame(request: PreviewRequest) -> dict:
    record = get_video(request.videoId)
    adapter_weight_path = get_adapter_weight_path(request.adapterId)
    preview_id = uuid.uuid4().hex
    preview_root = PREVIEW_DIR / preview_id
    original = preview_root / "original.png"
    enhanced = preview_root / "enhanced.png"
    adapter_preview: Path | None = None
    try:
        extract_frame(Path(record["path"]), request.seconds, original)
        if second_pass_enabled(request):
            adapter_preview = preview_root / "adapter_pass.png"
            engine.enhance_file(original, adapter_preview, request.to_hypir(adapter_weight_path))
            result = engine.enhance_file(adapter_preview, enhanced, request.to_base_refine_hypir())
        else:
            result = engine.enhance_file(original, enhanced, request.to_hypir(adapter_weight_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Preview failed: {exc}") from exc
    finally:
        if adapter_preview is not None:
            adapter_preview.unlink(missing_ok=True)

    return {
        "id": preview_id,
        "seconds": request.seconds,
        "originalUrl": f"/media/previews/{preview_id}/original.png",
        "enhancedUrl": f"/media/previews/{preview_id}/enhanced.png",
        "result": result,
        "passes": 2 if second_pass_enabled(request) else 1,
    }


def second_pass_enabled(request: ProcessSettings) -> bool:
    return request.secondPass == "base_after_adapter" and request.adapterId != "base"


def run_export(job_id: str, request: ExportRequest) -> None:
    try:
        record = get_video(request.videoId)
        source = Path(record["path"])
        metadata = record["metadata"]
        expected_count = int(metadata.get("frameCount") or 0)
        fps = float(metadata.get("fps") or 30.0)
        cache_key = enhancement_cache_key(record, request)
        output_path = export_output_path(record, cache_key, request)
        if job_id in cancelled_jobs:
            update_job(job_id, status="cancelled", progress=0, message="Stopped before export started")
            cancelled_jobs.discard(job_id)
            return
        if output_path.exists() and output_path.stat().st_size > 0:
            update_job(
                job_id,
                status="done",
                progress=1.0,
                message="Done. Reused cached H.264 output",
                etaSeconds=0,
                framesDone=expected_count,
                framesTotal=expected_count,
                partialFramesReady=expected_count,
                cacheKey=cache_key,
                outputUrl=f"/media/exports/{output_path.name}",
                outputPath=str(output_path),
            )
            return

        cache_root = CACHE_DIR / cache_key
        raw_frames = cache_root / "frames"
        enhanced_frames = cache_root / "enhanced"
        enhanced_frames.mkdir(parents=True, exist_ok=True)
        write_cache_manifest(cache_root, record, request, "running")

        update_job(
            job_id,
            status="running",
            progress=0.02,
            message="Preparing source frames",
            cacheKey=cache_key,
        )
        if job_id in cancelled_jobs:
            update_job(job_id, status="cancelled", progress=0, message="Stopped before export started")
            cancelled_jobs.discard(job_id)
            return

        frames = reusable_frames(source, raw_frames, expected_count)
        total = len(frames)
        adapter_weight_path = get_adapter_weight_path(request.adapterId)
        adapter_settings = request.to_hypir(adapter_weight_path)
        base_refine_settings = request.to_base_refine_hypir()
        use_second_pass = second_pass_enabled(request)
        adapter_pass_frames = cache_root / "adapter_pass"
        if use_second_pass:
            adapter_pass_frames.mkdir(parents=True, exist_ok=True)
        outputs = [
            (frame, enhanced_frames / frame.name, valid_image(enhanced_frames / frame.name))
            for frame in frames
        ]
        missing_total = sum(1 for _, _, cached in outputs if not cached)
        cache_hits = total - missing_total
        cache_misses_done = 0
        partial_frames_ready = len(contiguous_ready_frames(enhanced_frames))
        generation_elapsed = 0.0
        latest_frame_seq = 0
        temporal_enabled = temporal_mode_enabled(request.temporalConsistency)
        previous_source_frame: Path | None = None
        previous_enhanced_frame: Path | None = None

        if use_second_pass:
            stage_inputs = [
                (index, frame, adapter_pass_frames / frame.name)
                for index, (frame, _output, cached) in enumerate(outputs, start=1)
                if not cached
            ]
            stage_missing = [
                (index, frame, adapter_output)
                for index, frame, adapter_output in stage_inputs
                if not valid_image(adapter_output)
            ]
            for stage_index, (frame_index, frame, adapter_output) in enumerate(stage_missing, start=1):
                if job_id in cancelled_jobs:
                    write_cache_manifest(cache_root, record, request, "cancelled")
                    update_job(
                        job_id,
                        status="cancelled",
                        progress=0.05 + (stage_index - 1) / max(len(stage_missing), 1) * 0.40,
                        message=f"Stopped during adapter pass. Cached {cache_misses_done} / {total} final frames.",
                        framesDone=cache_misses_done,
                        framesTotal=total,
                        etaSeconds=None,
                    )
                    cancelled_jobs.discard(job_id)
                    return

                update_job(
                    job_id,
                    progress=0.05 + (stage_index - 1) / max(len(stage_missing), 1) * 0.40,
                    message=f"Adapter pass frame {frame_index} / {total}",
                    etaSeconds=None,
                    framesDone=stage_index - 1,
                    framesTotal=max(len(stage_missing), 1),
                    currentFrameIndex=frame_index,
                    currentFrameSeconds=(frame_index - 1) / fps,
                    currentOriginalUrl=media_url(frame),
                    currentEnhancedUrl=media_url(adapter_output) if valid_image(adapter_output) else None,
                    partialFramesReady=partial_frames_ready,
                    cacheHits=cache_hits,
                    cacheMisses=cache_misses_done,
                )
                frame_started = time.monotonic()
                engine.enhance_file(frame, adapter_output, adapter_settings)
                generation_elapsed += time.monotonic() - frame_started

            for index, (frame, output, cached) in enumerate(outputs, start=1):
                if job_id in cancelled_jobs:
                    write_cache_manifest(cache_root, record, request, "cancelled")
                    update_job(
                        job_id,
                        status="cancelled",
                        progress=0.45 + (index - 1) / max(total, 1) * 0.44,
                        message=f"Stopped. Cached {index - 1} / {total} enhanced frames.",
                        framesDone=index - 1,
                        framesTotal=total,
                        etaSeconds=None,
                    )
                    cancelled_jobs.discard(job_id)
                    return

                adapter_output = adapter_pass_frames / frame.name
                remaining_misses = max(0, missing_total - cache_misses_done)
                update_job(
                    job_id,
                    progress=0.45 + (index - 1) / max(total, 1) * 0.44,
                    message=(
                        f"Using cached frame {index} / {total}"
                        if cached else f"Base pass frame {index} / {total}"
                    ),
                    etaSeconds=eta_seconds(generation_elapsed, cache_misses_done, remaining_misses),
                    framesDone=index - 1,
                    framesTotal=total,
                    currentFrameIndex=index,
                    currentFrameSeconds=(index - 1) / fps,
                    currentOriginalUrl=media_url(frame),
                    currentEnhancedUrl=media_url(output) if cached else None,
                    cacheHits=cache_hits,
                    cacheMisses=cache_misses_done,
                )
                if not cached:
                    if not valid_image(adapter_output):
                        engine.enhance_file(frame, adapter_output, adapter_settings)
                    frame_started = time.monotonic()
                    if temporal_enabled and previous_source_frame and previous_enhanced_frame:
                        raw_output = output.with_name(f"{output.stem}.base_tmp{output.suffix}")
                        try:
                            engine.enhance_file(adapter_output, raw_output, base_refine_settings)
                            stabilize_frame(
                                previous_source_path=previous_source_frame,
                                current_source_path=frame,
                                previous_enhanced_path=previous_enhanced_frame,
                                current_enhanced_path=raw_output,
                                output_path=output,
                                mode=request.temporalConsistency,
                            )
                        finally:
                            raw_output.unlink(missing_ok=True)
                    else:
                        engine.enhance_file(adapter_output, output, base_refine_settings)
                    generation_elapsed += time.monotonic() - frame_started
                    cache_misses_done += 1
                    frame_event = add_frame_event(
                        job_id,
                        frame_index=index,
                        frames_total=total,
                        seconds=(index - 1) / fps,
                        original_path=frame,
                        enhanced_path=output,
                        cached=False,
                    )
                    latest_frame_seq = frame_event.seq

                frame_seconds = (index - 1) / fps
                remaining_misses = max(0, missing_total - cache_misses_done)
                partial_frames_ready = max(partial_frames_ready, index)
                update_job(
                    job_id,
                    progress=0.45 + index / max(total, 1) * 0.44,
                    message=f"Frame {index} / {total} ready",
                    etaSeconds=eta_seconds(generation_elapsed, cache_misses_done, remaining_misses),
                    framesDone=index,
                    framesTotal=total,
                    currentFrameIndex=index,
                    currentFrameSeconds=frame_seconds,
                    currentOriginalUrl=media_url(frame),
                    currentEnhancedUrl=media_url(output),
                    latestFrameSeq=latest_frame_seq,
                    partialFramesReady=partial_frames_ready,
                    cacheHits=cache_hits,
                    cacheMisses=cache_misses_done,
                )
                previous_source_frame = frame
                previous_enhanced_frame = output
        else:
            for index, (frame, output, cached) in enumerate(outputs, start=1):
                if job_id in cancelled_jobs:
                    write_cache_manifest(cache_root, record, request, "cancelled")
                    update_job(
                        job_id,
                        status="cancelled",
                        progress=0.05 + (index - 1) / max(total, 1) * 0.84,
                        message=f"Stopped. Cached {index - 1} / {total} enhanced frames.",
                        framesDone=index - 1,
                        framesTotal=total,
                        etaSeconds=None,
                    )
                    cancelled_jobs.discard(job_id)
                    return

                remaining_misses = max(0, missing_total - cache_misses_done)
                update_job(
                    job_id,
                    progress=0.05 + (index - 1) / max(total, 1) * 0.84,
                    message=(
                        f"Using cached frame {index} / {total}"
                        if cached else f"Enhancing frame {index} / {total}"
                    ),
                    etaSeconds=eta_seconds(generation_elapsed, cache_misses_done, remaining_misses),
                    framesDone=index - 1,
                    framesTotal=total,
                    currentFrameIndex=index,
                    currentFrameSeconds=(index - 1) / fps,
                    currentOriginalUrl=media_url(frame),
                    currentEnhancedUrl=media_url(output) if cached else None,
                    cacheHits=cache_hits,
                    cacheMisses=cache_misses_done,
                )
                if not cached:
                    frame_started = time.monotonic()
                    if temporal_enabled and previous_source_frame and previous_enhanced_frame:
                        raw_output = output.with_name(f"{output.stem}.hypir_tmp{output.suffix}")
                        try:
                            engine.enhance_file(frame, raw_output, adapter_settings)
                            stabilize_frame(
                                previous_source_path=previous_source_frame,
                                current_source_path=frame,
                                previous_enhanced_path=previous_enhanced_frame,
                                current_enhanced_path=raw_output,
                                output_path=output,
                                mode=request.temporalConsistency,
                            )
                        finally:
                            raw_output.unlink(missing_ok=True)
                    else:
                        engine.enhance_file(frame, output, adapter_settings)
                    generation_elapsed += time.monotonic() - frame_started
                    cache_misses_done += 1
                    frame_event = add_frame_event(
                        job_id,
                        frame_index=index,
                        frames_total=total,
                        seconds=(index - 1) / fps,
                        original_path=frame,
                        enhanced_path=output,
                        cached=False,
                    )
                    latest_frame_seq = frame_event.seq

                frame_seconds = (index - 1) / fps
                remaining_misses = max(0, missing_total - cache_misses_done)
                partial_frames_ready = max(partial_frames_ready, index)
                update_job(
                    job_id,
                    progress=0.05 + index / max(total, 1) * 0.84,
                    message=f"Frame {index} / {total} ready",
                    etaSeconds=eta_seconds(generation_elapsed, cache_misses_done, remaining_misses),
                    framesDone=index,
                    framesTotal=total,
                    currentFrameIndex=index,
                    currentFrameSeconds=frame_seconds,
                    currentOriginalUrl=media_url(frame),
                    currentEnhancedUrl=media_url(output),
                    latestFrameSeq=latest_frame_seq,
                    partialFramesReady=partial_frames_ready,
                    cacheHits=cache_hits,
                    cacheMisses=cache_misses_done,
                )
                previous_source_frame = frame
                previous_enhanced_frame = output

        write_cache_manifest(cache_root, record, request, "complete")
        update_job(job_id, progress=0.92, message="Encoding H.264 video", etaSeconds=None)
        selected_encoder = encode_video(
            enhanced_frames,
            source,
            output_path,
            fps=fps,
            crf=request.crf,
            encoder=request.encoder,
        )
        cleanup_message = "Scheduled intermediate cleanup"
        try:
            cleanup_delay = min(900, max(120, int(total * 0.04)))
            cleanup_usage = schedule_generated_path_cleanup(cache_root, delay_seconds=cleanup_delay)
            cleanup_message = f"Scheduled cleanup of {cleanup_usage['files']} intermediate files"
        except Exception as exc:
            cleanup_message = f"Intermediate cleanup failed: {exc}"
        update_job(
            job_id,
            status="done",
            progress=1.0,
            message=f"Done. Encoded with {selected_encoder}. {cleanup_message}",
            etaSeconds=0,
            framesDone=total,
            framesTotal=total,
            partialFramesReady=total,
            outputUrl=f"/media/exports/{output_path.name}",
            outputPath=str(output_path),
        )
    except Exception as exc:
        update_job(
            job_id,
            status="error",
            progress=0,
            message="Export failed",
            etaSeconds=None,
            error=f"{exc}\n{traceback.format_exc()}",
        )
    finally:
        cancelled_jobs.discard(job_id)


@app.post("/api/export")
def export_video(request: ExportRequest, background_tasks: BackgroundTasks) -> dict:
    get_video(request.videoId)
    job_id = uuid.uuid4().hex
    stamp = now_iso()
    with jobs_lock:
        jobs[job_id] = JobState(
            id=job_id,
            kind="export",
            status="queued",
            progress=0,
            message="Queued",
            videoId=request.videoId,
            startedAt=stamp,
            updatedAt=stamp,
        )
        job_frame_events[job_id] = []
    background_tasks.add_task(run_export, job_id, request)
    return jobs[job_id].model_dump()


@app.get("/api/jobs")
def list_jobs() -> dict:
    with jobs_lock:
        ordered = sorted(jobs.values(), key=lambda job: job.startedAt, reverse=True)
    return {"jobs": [job.model_dump() for job in ordered]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.model_dump()


@app.post("/api/jobs/{job_id}/partial-export")
def export_partial_video(job_id: str, request: PartialExportRequest) -> dict:
    partial_token = uuid.uuid4().hex
    partial_root = PARTIAL_DIR / partial_token
    partial_frames = partial_root / "frames"
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        job_snapshot = job.model_dump()
        partial_exports_active.add(partial_token)

    try:
        if job_snapshot["kind"] != "export":
            raise HTTPException(status_code=400, detail="Partial video is only available for export jobs")
        if job_snapshot["status"] not in {"running", "done"}:
            raise HTTPException(status_code=409, detail="Export has not started producing frames yet")
        if not job_snapshot.get("videoId"):
            raise HTTPException(status_code=400, detail="Export job is missing its source video")
        cache_key = job_snapshot.get("cacheKey")
        if not cache_key:
            raise HTTPException(status_code=409, detail="Export is still preparing the frame cache")

        record = get_video(job_snapshot["videoId"])
        source = Path(record["path"])
        metadata = record["metadata"]
        fps = float(metadata.get("fps") or 30.0)
        enhanced_frames = CACHE_DIR / cache_key / "enhanced"
        ready_frames = contiguous_ready_frames(enhanced_frames)
        frame_count = len(ready_frames)
        if frame_count <= 0:
            raise HTTPException(status_code=409, detail="No enhanced frames are ready yet")

        snapshot_partial_frames(ready_frames, partial_frames)
        output_path = partial_output_path(record, cache_key, frame_count, request)
        try:
            selected_encoder = encode_video(
                partial_frames,
                source,
                output_path,
                fps=fps,
                crf=request.crf,
                encoder=request.encoder,
                frame_count=frame_count,
            )
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"Partial export failed: {exc}") from exc

        return {
            "outputUrl": f"/media/exports/{output_path.name}",
            "outputPath": str(output_path),
            "framesDone": frame_count,
            "framesTotal": job_snapshot.get("framesTotal") or int(metadata.get("frameCount") or 0),
            "durationSeconds": frame_count / fps if fps > 0 else 0,
            "encoder": selected_encoder,
        }
    finally:
        remove_generated_path(partial_root, recreate=False)
        with jobs_lock:
            partial_exports_active.discard(partial_token)


@app.get("/api/jobs/{job_id}/frames")
def get_job_frames(
    job_id: str,
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=240, ge=1, le=500),
) -> dict:
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        events = job_frame_events.get(job_id, [])
        selected = [event for event in events if event.seq > after][:limit]
        latest_seq = events[-1].seq if events else 0
    return {
        "frames": [event.model_dump() for event in selected],
        "nextAfter": selected[-1].seq if selected else after,
        "latestSeq": latest_seq,
        "hasMore": bool(selected and selected[-1].seq < latest_seq),
    }


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status in {"done", "error", "cancelled"}:
        return job.model_dump()
    cancelled_jobs.add(job_id)
    process = job_processes.get(job_id)
    if process and process.poll() is None:
        process.terminate()
    update_job(
        job_id,
        message="Stopping film adapter training" if job.kind == "adapter" else "Stopping after the current frame",
    )
    return jobs[job_id].model_dump()


app.mount("/media", StaticFiles(directory=WORK_DIR), name="media")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
