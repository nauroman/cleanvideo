from __future__ import annotations

import json
import hashlib
import shutil
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from .hypir_engine import HypirSettings, engine
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
WORK_DIR = ROOT / "work"
UPLOAD_DIR = WORK_DIR / "uploads"
PREVIEW_DIR = WORK_DIR / "previews"
EXPORT_DIR = WORK_DIR / "exports"
JOB_DIR = WORK_DIR / "jobs"
CACHE_DIR = WORK_DIR / "cache"
for directory in [UPLOAD_DIR, PREVIEW_DIR, EXPORT_DIR, JOB_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


class ProcessSettings(BaseModel):
    engine: Literal["hypir"] = "hypir"
    prompt: str = ""
    scaleBy: Literal["factor", "longest_side"] = "factor"
    upscale: int = Field(default=1, ge=1, le=8)
    targetLongestSide: int | None = Field(default=None, ge=256, le=8192)
    patchSize: int = Field(default=512, ge=512, le=1024)
    stride: int = Field(default=256, ge=128, le=1024)
    seed: int = Field(default=231, ge=-1)
    device: Literal["cuda"] = "cuda"

    def to_hypir(self) -> HypirSettings:
        return HypirSettings(
            prompt=self.prompt.strip(),
            scale_by=self.scaleBy,
            upscale=self.upscale,
            target_longest_side=self.targetLongestSide,
            patch_size=self.patchSize,
            stride=self.stride,
            seed=self.seed,
            device=self.device,
        )


class PreviewRequest(ProcessSettings):
    videoId: str
    seconds: float = Field(default=0.0, ge=0)


class ExportRequest(ProcessSettings):
    videoId: str
    crf: int = Field(default=18, ge=12, le=32)
    encoder: Literal["auto", "h264_nvenc", "libx264"] = "auto"


class JobState(BaseModel):
    id: str
    kind: str
    status: Literal["queued", "running", "done", "error", "cancelled"]
    progress: float
    message: str
    etaSeconds: float | None = None
    framesDone: int = 0
    framesTotal: int = 0
    currentFrameIndex: int | None = None
    currentFrameSeconds: float | None = None
    currentOriginalUrl: str | None = None
    currentEnhancedUrl: str | None = None
    cacheKey: str | None = None
    cacheHits: int = 0
    cacheMisses: int = 0
    outputUrl: str | None = None
    outputPath: str | None = None
    error: str | None = None
    startedAt: str
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
jobs: dict[str, JobState] = {}
jobs_lock = threading.Lock()
cancelled_jobs: set[str] = set()


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


def update_job(job_id: str, **changes) -> None:
    with jobs_lock:
        job = jobs[job_id]
        data = job.model_dump()
        data.update(changes)
        data["updatedAt"] = now_iso()
        jobs[job_id] = JobState(**data)


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
    return request.model_dump(exclude={"crf", "encoder"}, mode="json")


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


def eta_seconds(started: float, completed_misses: int, remaining_misses: int) -> float | None:
    if remaining_misses <= 0:
        return 0.0
    if completed_misses <= 0:
        return None
    elapsed = max(0.0, time.monotonic() - started)
    return elapsed / completed_misses * remaining_misses


@app.on_event("startup")
def startup() -> None:
    load_video_records()


@app.get("/api/status")
def status() -> dict:
    return {
        "app": "CleanVideo",
        "hypir": engine.status(),
        "ffmpeg": True,
        "nvenc": h264_nvenc_available(),
        "videos": len(videos),
        "jobs": len(jobs),
    }


@app.get("/api/videos")
def list_videos() -> dict:
    return {"videos": list(videos.values())}


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
    preview_id = uuid.uuid4().hex
    preview_root = PREVIEW_DIR / preview_id
    original = preview_root / "original.png"
    enhanced = preview_root / "enhanced.png"
    try:
        extract_frame(Path(record["path"]), request.seconds, original)
        result = engine.enhance_file(original, enhanced, request.to_hypir())
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
        cache_key = enhancement_cache_key(record, request)
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

        expected_count = int(metadata.get("frameCount") or 0)
        frames = reusable_frames(source, raw_frames, expected_count)
        total = len(frames)
        fps = float(metadata.get("fps") or 30.0)
        settings = request.to_hypir()
        outputs = [(frame, enhanced_frames / frame.name) for frame in frames]
        missing_total = sum(1 for _, output in outputs if not valid_image(output))
        cache_hits = total - missing_total
        cache_misses_done = 0
        started = time.monotonic()

        for index, (frame, output) in enumerate(outputs, start=1):
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

            cached = valid_image(output)
            remaining_misses = max(0, missing_total - cache_misses_done)
            update_job(
                job_id,
                progress=0.05 + (index - 1) / max(total, 1) * 0.84,
                message=(
                    f"Using cached frame {index} / {total}"
                    if cached else f"Enhancing frame {index} / {total}"
                ),
                etaSeconds=eta_seconds(started, cache_misses_done, remaining_misses),
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
                pass
            else:
                engine.enhance_file(frame, output, settings)
                cache_misses_done += 1

            remaining_misses = max(0, missing_total - cache_misses_done)
            update_job(
                job_id,
                progress=0.05 + index / max(total, 1) * 0.84,
                message=f"Frame {index} / {total} ready",
                etaSeconds=eta_seconds(started, cache_misses_done, remaining_misses),
                framesDone=index,
                framesTotal=total,
                currentFrameIndex=index,
                currentFrameSeconds=(index - 1) / fps,
                currentOriginalUrl=media_url(frame),
                currentEnhancedUrl=media_url(output),
                cacheHits=cache_hits,
                cacheMisses=cache_misses_done,
            )

        write_cache_manifest(cache_root, record, request, "complete")
        output_path = export_output_path(record, cache_key, request)
        if output_path.exists() and output_path.stat().st_size > 0:
            selected_encoder = "cached H.264"
            update_job(job_id, progress=0.96, message="Using cached H.264 output", etaSeconds=0)
        else:
            update_job(job_id, progress=0.92, message="Encoding H.264 video", etaSeconds=None)
            selected_encoder = encode_video(
                enhanced_frames,
                source,
                output_path,
                fps=fps,
                crf=request.crf,
                encoder=request.encoder,
            )
        update_job(
            job_id,
            status="done",
            progress=1.0,
            message=f"Done. Encoded with {selected_encoder}",
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
    jobs[job_id] = JobState(
        id=job_id,
        kind="export",
        status="queued",
        progress=0,
        message="Queued",
        startedAt=stamp,
        updatedAt=stamp,
    )
    background_tasks.add_task(run_export, job_id, request)
    return jobs[job_id].model_dump()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.model_dump()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status in {"done", "error", "cancelled"}:
        return job.model_dump()
    cancelled_jobs.add(job_id)
    update_job(job_id, message="Stopping after the current frame")
    return jobs[job_id].model_dump()


app.mount("/media", StaticFiles(directory=WORK_DIR), name="media")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
