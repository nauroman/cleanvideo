from __future__ import annotations

import json
import hashlib
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

from .film_adapter import build_adapter_dataset
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
ADAPTER_DIR = WORK_DIR / "adapters"
for directory in [UPLOAD_DIR, PREVIEW_DIR, EXPORT_DIR, JOB_DIR, CACHE_DIR, ADAPTER_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
GENERATED_DIRS = [PREVIEW_DIR, CACHE_DIR, JOB_DIR, EXPORT_DIR]
APP_BUILD = "2026-06-11-film-adapter-v1"


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


class PreviewRequest(ProcessSettings):
    videoId: str
    seconds: float = Field(default=0.0, ge=0)


class ExportRequest(ProcessSettings):
    videoId: str
    crf: int = Field(default=18, ge=12, le=32)
    encoder: Literal["auto", "h264_nvenc", "libx264"] = "auto"


class AdapterTrainRequest(BaseModel):
    videoId: str
    prompt: str = "film-specific restoration, natural detail, consistent texture"
    maxFrames: int = Field(default=32, ge=4, le=240)
    patchesPerFrame: int = Field(default=3, ge=1, le=9)
    maxTrainSteps: int = Field(default=300, ge=20, le=3000)


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
    return [base, *ordered]


def get_adapter_weight_path(adapter_id: str) -> Path | None:
    if adapter_id == "base":
        return None
    record = adapters.get(adapter_id)
    if not record:
        raise HTTPException(status_code=404, detail="Film adapter not found")
    weight_path = Path(record["weightPath"])
    if not weight_path.exists():
        raise HTTPException(status_code=404, detail="Film adapter weights are missing")
    return weight_path


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


def latest_checkpoint_weight(output_dir: Path) -> Path | None:
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
            checkpoints.append((step, weight_path))
    if not checkpoints:
        return None
    return sorted(checkpoints, key=lambda item: item[0])[-1][1]


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
        update_job(
            job_id,
            status="running",
            progress=0.04,
            message="Sampling film frames for adapter training",
            videoId=request.videoId,
        )
        dataset = build_adapter_dataset(
            video_path=source,
            duration_seconds=float(metadata.get("duration") or 0),
            adapter_root=adapter_root,
            base_model_path=engine.base_model_path,
            prompt=prompt,
            max_frames=request.maxFrames,
            patches_per_frame=request.patchesPerFrame,
            max_train_steps=request.maxTrainSteps,
        )
        if job_id in cancelled_jobs:
            update_job(job_id, status="cancelled", progress=0.12, message="Film adapter training cancelled")
            cancelled_jobs.discard(job_id)
            return

        update_job(
            job_id,
            progress=0.16,
            message=f"Training film adapter on {dataset.patches} patches from {dataset.frames} frames",
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
        with log_path.open("w", encoding="utf-8", errors="replace") as log_fp:
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
                    checkpoint = latest_checkpoint_weight(dataset.output_dir)
                    progress = 0.24 if checkpoint is None else 0.55
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
        if return_code != 0:
            raise RuntimeError(f"HYPIR adapter training failed with exit code {return_code}. See {log_path}")

        weight_path = latest_checkpoint_weight(dataset.output_dir)
        if not weight_path:
            raise RuntimeError(f"Training finished, but no checkpoint state_dict.pth was found under {dataset.output_dir}")
        adapter_name = f"{Path(record['name']).stem} Film Adapter"
        adapter_record = {
            "id": adapter_id,
            "name": adapter_name,
            "status": "ready",
            "videoId": request.videoId,
            "videoName": record["name"],
            "weightPath": str(weight_path),
            "rootPath": str(adapter_root),
            "prompt": prompt,
            "frames": dataset.frames,
            "patches": dataset.patches,
            "maxTrainSteps": request.maxTrainSteps,
            "createdAt": now_iso(),
        }
        adapters[adapter_id] = adapter_record
        save_adapter_record(adapter_record)
        update_job(
            job_id,
            status="done",
            progress=1.0,
            message=f"Film adapter ready: {adapter_name}",
            adapterId=adapter_id,
            adapterName=adapter_name,
            outputPath=str(weight_path),
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
    if active_jobs:
        raise HTTPException(status_code=409, detail="Cannot clean generated files while an export is running")
    try:
        result = clear_generated_dirs()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {exc}") from exc
    with jobs_lock:
        jobs.clear()
        job_frame_events.clear()
    cancelled_jobs.clear()
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
    try:
        extract_frame(Path(record["path"]), request.seconds, original)
        result = engine.enhance_file(original, enhanced, request.to_hypir(adapter_weight_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Preview failed: {exc}") from exc

    return {
        "id": preview_id,
        "seconds": request.seconds,
        "originalUrl": f"/media/previews/{preview_id}/original.png",
        "enhancedUrl": f"/media/previews/{preview_id}/enhanced.png",
        "result": result,
    }


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
        settings = request.to_hypir(adapter_weight_path)
        outputs = [
            (frame, enhanced_frames / frame.name, valid_image(enhanced_frames / frame.name))
            for frame in frames
        ]
        missing_total = sum(1 for _, _, cached in outputs if not cached)
        cache_hits = total - missing_total
        cache_misses_done = 0
        generation_elapsed = 0.0
        latest_frame_seq = 0
        temporal_enabled = temporal_mode_enabled(request.temporalConsistency)
        previous_source_frame: Path | None = None
        previous_enhanced_frame: Path | None = None

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
            if cached:
                frame_event = None
            else:
                frame_started = time.monotonic()
                if temporal_enabled and previous_source_frame and previous_enhanced_frame:
                    raw_output = output.with_name(f"{output.stem}.hypir_tmp{output.suffix}")
                    try:
                        engine.enhance_file(frame, raw_output, settings)
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
                    engine.enhance_file(frame, output, settings)
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
