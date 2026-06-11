const state = {
  video: null,
  jobTimer: null,
  currentJobId: null,
  previewTimer: null,
  previewInFlight: false,
  previewDirty: false,
  previewVersion: 0,
  exportInFlight: false,
  liveOriginalUrl: null,
  liveEnhancedUrl: null,
  frameEventQueue: [],
  frameEventsAfter: 0,
  frameEventsInFlight: false,
  framePlaybackTimer: null,
  lastDisplayedFrameSeq: 0,
};

const settingsStorageKey = "cleanvideo.session.v1";

const resolutionPresets = {
  preset_720p: 1280,
  preset_fullhd: 1920,
  preset_4k: 3840,
  preset_8k: 7680,
};

const els = {
  videoInput: document.querySelector("#videoInput"),
  videoName: document.querySelector("#videoName"),
  videoMeta: document.querySelector("#videoMeta"),
  sourceVideo: document.querySelector("#sourceVideo"),
  originalPreview: document.querySelector("#originalPreview"),
  enhancedPreview: document.querySelector("#enhancedPreview"),
  originalEmpty: document.querySelector("#originalEmpty"),
  enhancedEmpty: document.querySelector("#enhancedEmpty"),
  originalInfo: document.querySelector("#originalInfo"),
  enhancedInfo: document.querySelector("#enhancedInfo"),
  timeline: document.querySelector("#timeline"),
  currentTime: document.querySelector("#currentTime"),
  durationTime: document.querySelector("#durationTime"),
  exportButton: document.querySelector("#exportButton"),
  cancelExportButton: document.querySelector("#cancelExportButton"),
  openOutputFolderButton: document.querySelector("#openOutputFolderButton"),
  clearGeneratedButton: document.querySelector("#clearGeneratedButton"),
  activityText: document.querySelector("#activityText"),
  frameProgressText: document.querySelector("#frameProgressText"),
  etaText: document.querySelector("#etaText"),
  jobProgress: document.querySelector("#jobProgress"),
  downloadLink: document.querySelector("#downloadLink"),
  refreshStatusButton: document.querySelector("#refreshStatusButton"),
  gpuStatus: document.querySelector("#gpuStatus"),
  nvencStatus: document.querySelector("#nvencStatus"),
  modelStatus: document.querySelector("#modelStatus"),
  scaleBy: document.querySelector("#scaleBy"),
  upscale: document.querySelector("#upscale"),
  targetLongestSide: document.querySelector("#targetLongestSide"),
  targetField: document.querySelector("#targetField"),
  upscaleField: document.querySelector("#upscaleField"),
  patchSize: document.querySelector("#patchSize"),
  stride: document.querySelector("#stride"),
  temporalConsistency: document.querySelector("#temporalConsistency"),
  seed: document.querySelector("#seed"),
  prompt: document.querySelector("#prompt"),
  encoder: document.querySelector("#encoder"),
  crf: document.querySelector("#crf"),
  crfValue: document.querySelector("#crfValue"),
};

const tooltip = document.createElement("div");
tooltip.className = "tooltip-popover";
tooltip.setAttribute("role", "tooltip");
document.body.appendChild(tooltip);

function formatTime(seconds) {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const minutes = Math.floor(safe / 60);
  const secs = Math.floor(safe % 60);
  const ms = Math.floor((safe - Math.floor(safe)) * 1000);
  return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}.${String(ms).padStart(3, "0")}`;
}

function hideTooltip() {
  tooltip.classList.remove("visible");
  tooltip.textContent = "";
  document.querySelectorAll(".help-button.tooltip-active").forEach((button) => {
    button.classList.remove("tooltip-active");
  });
}

function showTooltip(button) {
  const text = button.dataset.tooltip;
  if (!text) return;

  document.querySelectorAll(".help-button.tooltip-active").forEach((activeButton) => {
    if (activeButton !== button) activeButton.classList.remove("tooltip-active");
  });
  button.classList.add("tooltip-active");
  tooltip.textContent = text;
  tooltip.style.left = "0px";
  tooltip.style.top = "0px";
  tooltip.classList.add("visible");

  const buttonRect = button.getBoundingClientRect();
  const tooltipRect = tooltip.getBoundingClientRect();
  const margin = 12;
  const left = Math.min(
    window.innerWidth - tooltipRect.width - margin,
    Math.max(margin, buttonRect.left)
  );
  let top = buttonRect.bottom + 8;
  if (top + tooltipRect.height > window.innerHeight - margin) {
    top = Math.max(margin, buttonRect.top - tooltipRect.height - 8);
  }

  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${top}px`;
}

function setupTooltips() {
  document.querySelectorAll(".help-button").forEach((button) => {
    button.addEventListener("mouseenter", () => showTooltip(button));
    button.addEventListener("mouseleave", hideTooltip);
    button.addEventListener("focus", () => showTooltip(button));
    button.addEventListener("blur", hideTooltip);
    button.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        hideTooltip();
        button.blur();
      }
    });
  });
  window.addEventListener("scroll", hideTooltip, true);
  window.addEventListener("resize", hideTooltip);
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return "--";
  const safe = Math.max(0, Math.round(seconds));
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const secs = safe % 60;
  if (hours > 0) {
    return `${hours}h ${String(minutes).padStart(2, "0")}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${String(secs).padStart(2, "0")}s`;
  }
  return `${secs}s`;
}

function formatBytes(bytes) {
  const safe = Number.isFinite(bytes) ? Math.max(0, bytes) : 0;
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = safe;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  const digits = value >= 10 || unitIndex === 0 ? 0 : 1;
  return `${value.toFixed(digits)} ${units[unitIndex]}`;
}

function setActivity(message, progress = null) {
  els.activityText.textContent = message;
  if (progress !== null) {
    els.jobProgress.style.width = `${Math.max(0, Math.min(1, progress)) * 100}%`;
  }
}

function collectUiSettings() {
  return {
    scaleBy: els.scaleBy.value,
    upscale: els.upscale.value,
    targetLongestSide: els.targetLongestSide.value,
    patchSize: els.patchSize.value,
    stride: els.stride.value,
    temporalConsistency: els.temporalConsistency.value,
    seed: els.seed.value,
    prompt: els.prompt.value,
    encoder: els.encoder.value,
    crf: els.crf.value,
  };
}

function saveLocalState() {
  const payload = {
    settings: collectUiSettings(),
    videoId: state.video?.id ?? null,
    currentJobId: state.exportInFlight ? state.currentJobId : null,
  };
  localStorage.setItem(settingsStorageKey, JSON.stringify(payload));
}

function readLocalState() {
  try {
    return JSON.parse(localStorage.getItem(settingsStorageKey) || "{}");
  } catch {
    return {};
  }
}

function applySavedSettings(settings = {}) {
  const entries = [
    ["scaleBy", els.scaleBy],
    ["upscale", els.upscale],
    ["targetLongestSide", els.targetLongestSide],
    ["patchSize", els.patchSize],
    ["stride", els.stride],
    ["temporalConsistency", els.temporalConsistency],
    ["seed", els.seed],
    ["prompt", els.prompt],
    ["encoder", els.encoder],
    ["crf", els.crf],
  ];
  for (const [key, element] of entries) {
    if (settings[key] !== undefined && element) {
      element.value = settings[key];
    }
  }
  els.crfValue.textContent = els.crf.value;
  updateScaleMode();
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      detail = await response.text();
    }
    throw new Error(detail);
  }
  return response.json();
}

function collectSettings() {
  const selectedScale = els.scaleBy.value;
  const presetLongestSide = resolutionPresets[selectedScale] ?? null;
  const scaleBy = presetLongestSide || selectedScale === "longest_side" ? "longest_side" : "factor";
  const patchSize = Number(els.patchSize.value);
  let stride = Number(els.stride.value);
  if (stride > patchSize) {
    stride = patchSize;
    els.stride.value = String(stride);
  }
  return {
    engine: "hypir",
    prompt: els.prompt.value,
    scaleBy,
    upscale: Number(els.upscale.value),
    targetLongestSide: scaleBy === "longest_side" ? presetLongestSide || Number(els.targetLongestSide.value) : null,
    patchSize,
    stride,
    temporalConsistency: els.temporalConsistency.value,
    seed: Number(els.seed.value),
    device: "cuda",
  };
}

function updateScaleMode() {
  const selectedScale = els.scaleBy.value;
  const customTargetMode = selectedScale === "longest_side";
  const fixedTargetMode = Boolean(resolutionPresets[selectedScale]);
  els.targetField.classList.toggle("hidden", !customTargetMode);
  els.upscaleField.classList.toggle("hidden", customTargetMode || fixedTargetMode);
}

function resetGeneratedUi() {
  clearFramePlayback();
  state.currentJobId = null;
  state.liveOriginalUrl = null;
  state.liveEnhancedUrl = null;
  els.downloadLink.hidden = true;
  els.downloadLink.removeAttribute("href");
  els.originalPreview.hidden = true;
  els.originalPreview.removeAttribute("src");
  els.enhancedPreview.hidden = true;
  els.enhancedPreview.removeAttribute("src");
  els.enhancedEmpty.hidden = false;
  if (state.video) {
    els.sourceVideo.hidden = false;
    els.originalEmpty.hidden = true;
  } else {
    els.sourceVideo.hidden = true;
    els.originalEmpty.hidden = false;
  }
  els.originalInfo.textContent = "source frame";
  els.enhancedInfo.textContent = "HYPIR preview";
  els.frameProgressText.textContent = "Frames: --";
  els.etaText.textContent = "ETA: --";
  els.jobProgress.style.width = "0%";
}

function setVideo(record, { persist = true } = {}) {
  state.video = record;
  state.liveOriginalUrl = null;
  state.liveEnhancedUrl = null;
  const meta = record.metadata;
  els.videoName.textContent = record.name;
  els.videoMeta.textContent = `${meta.width}x${meta.height} | ${meta.fps.toFixed(3)} fps | ${formatTime(meta.duration)} | ${meta.codec || "video"}`;
  els.sourceVideo.src = record.url;
  els.sourceVideo.hidden = false;
  els.originalPreview.hidden = true;
  els.originalEmpty.hidden = true;
  els.enhancedPreview.hidden = true;
  els.enhancedEmpty.hidden = false;
  els.timeline.max = String(meta.duration || 0);
  els.timeline.value = "0";
  els.currentTime.textContent = formatTime(0);
  els.durationTime.textContent = formatTime(meta.duration || 0);
  els.frameProgressText.textContent = "Frames: --";
  els.etaText.textContent = "ETA: --";
  setExportEnabled(true);
  els.downloadLink.hidden = true;
  if (persist) saveLocalState();
  setActivity("Preparing preview", 0.1);
  scheduleAutoPreview({ delay: 200, reason: "new video" });
}

async function restoreSession() {
  const saved = readLocalState();
  applySavedSettings(saved.settings);
  let restoredVideo = false;
  try {
    if (saved.videoId) {
      const result = await api("/api/videos");
      const record = result.videos.find((video) => video.id === saved.videoId);
      if (record) {
        restoredVideo = true;
        setVideo(record, { persist: false });
      }
    }
    await restoreActiveExport(saved.currentJobId, { restoredVideo });
  } catch (error) {
    setActivity(`Could not restore session: ${error.message}`, 0);
  }
}

async function restoreActiveExport(savedJobId, { restoredVideo = false } = {}) {
  let job = null;
  if (savedJobId) {
    try {
      job = await api(`/api/jobs/${savedJobId}`);
    } catch {
      job = null;
    }
  }

  if (!job || !["queued", "running"].includes(job.status)) {
    try {
      const result = await api("/api/jobs");
      job = result.jobs.find((candidate) => (
        candidate.kind === "export" && ["queued", "running"].includes(candidate.status)
      ));
    } catch {
      job = null;
    }
  }

  if (!job) return;

  if (!state.video && job.videoId) {
    const result = await api("/api/videos");
    const record = result.videos.find((video) => video.id === job.videoId);
    if (record) {
      restoredVideo = true;
      setVideo(record, { persist: false });
    }
  }

  resumeExportJob(job, { restoredVideo });
}

function resumeExportJob(job, { restoredVideo = false } = {}) {
  clearFramePlayback();
  state.exportInFlight = true;
  state.previewDirty = false;
  state.currentJobId = job.id;
  setExportEnabled(false);
  if (restoredVideo) {
    window.clearTimeout(state.previewTimer);
  }
  els.downloadLink.hidden = true;
  updateJobUi(job);
  saveLocalState();
  fetchFrameEvents(job.id);
  pollJob(job.id);
}

async function refreshStatus() {
  try {
    const status = await api("/api/status");
    const gpu = status.hypir.gpu || "no cuda";
    els.gpuStatus.textContent = status.hypir.cudaAvailable ? gpu.replace("NVIDIA GeForce ", "") : "none";
    els.nvencStatus.textContent = status.nvenc ? "ready" : "missing";
    els.modelStatus.textContent = status.hypir.loaded ? "loaded" : "cold";
  } catch (error) {
    els.gpuStatus.textContent = "error";
    els.nvencStatus.textContent = "error";
    els.modelStatus.textContent = "error";
  }
}

async function uploadVideo(file) {
  const form = new FormData();
  form.append("file", file);
  setActivity("Uploading video", 0.05);
  const record = await api("/api/videos", {
    method: "POST",
    body: form,
  });
  setVideo(record);
}

function setExportEnabled(enabled) {
  els.exportButton.disabled = !enabled || !state.video || state.exportInFlight || state.previewInFlight;
  els.cancelExportButton.hidden = !state.exportInFlight;
  els.cancelExportButton.disabled = !state.exportInFlight;
  els.clearGeneratedButton.disabled = state.exportInFlight || state.previewInFlight;
}

function scheduleAutoPreview({ delay = 450, reason = "settings changed" } = {}) {
  if (!state.video || state.exportInFlight) return;
  state.previewDirty = true;
  window.clearTimeout(state.previewTimer);
  setActivity(`Preview queued: ${reason}`, 0.12);
  state.previewTimer = window.setTimeout(runAutoPreview, delay);
}

async function runAutoPreview() {
  if (!state.video) return;
  if (state.previewInFlight) return;
  state.previewDirty = false;
  state.previewInFlight = true;
  setExportEnabled(false);
  const version = ++state.previewVersion;
  setActivity("Enhancing preview", 0.2);
  try {
    const body = {
      videoId: state.video.id,
      seconds: Number(els.timeline.value),
      ...collectSettings(),
    };
    const preview = await api("/api/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (version !== state.previewVersion || state.previewDirty) return;
    els.sourceVideo.hidden = true;
    els.originalPreview.src = `${preview.originalUrl}?t=${Date.now()}`;
    els.enhancedPreview.src = `${preview.enhancedUrl}?t=${Date.now()}`;
    els.originalPreview.hidden = false;
    els.enhancedPreview.hidden = false;
    els.originalEmpty.hidden = true;
    els.enhancedEmpty.hidden = true;
    els.originalInfo.textContent = formatTime(preview.seconds);
    els.enhancedInfo.textContent = `${preview.result.width}x${preview.result.height} | seed ${preview.result.seed}`;
    setActivity("Preview ready", 1);
    await refreshStatus();
  } catch (error) {
    setActivity(`Preview failed: ${error.message}`, 0);
  } finally {
    state.previewInFlight = false;
    setExportEnabled(true);
    if (state.previewDirty && !state.exportInFlight) {
      scheduleAutoPreview({ delay: 50, reason: "latest change" });
    }
  }
}

async function startExport() {
  if (!state.video) return;
  window.clearTimeout(state.previewTimer);
  clearFramePlayback();
  state.exportInFlight = true;
  state.previewDirty = false;
  setExportEnabled(false);
  saveLocalState();
  els.downloadLink.hidden = true;
  els.frameProgressText.textContent = "Frames: preparing";
  els.etaText.textContent = "ETA: estimating";
  setActivity("Starting export", 0.02);
  try {
    const body = {
      videoId: state.video.id,
      crf: Number(els.crf.value),
      encoder: els.encoder.value,
      ...collectSettings(),
    };
    const job = await api("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    state.currentJobId = job.id;
    state.frameEventsAfter = 0;
    saveLocalState();
    pollJob(job.id);
  } catch (error) {
    setActivity(`Export failed: ${error.message}`, 0);
    state.exportInFlight = false;
    setExportEnabled(true);
  }
}

async function cancelExport() {
  if (!state.currentJobId) return;
  els.cancelExportButton.disabled = true;
  setActivity("Stopping after the current frame", null);
  try {
    await api(`/api/jobs/${state.currentJobId}/cancel`, { method: "POST" });
  } catch (error) {
    setActivity(`Stop failed: ${error.message}`, null);
    els.cancelExportButton.disabled = false;
  }
}

async function openOutputFolder() {
  els.openOutputFolderButton.disabled = true;
  try {
    const result = await api("/api/open-output-folder", { method: "POST" });
    setActivity(`Opened output folder: ${result.path}`, null);
  } catch (error) {
    setActivity(`Could not open output folder: ${error.message}`, null);
  } finally {
    els.openOutputFolderButton.disabled = false;
  }
}

async function clearGeneratedFiles() {
  if (state.exportInFlight || state.previewInFlight) return;
  const confirmed = window.confirm(
    "Delete all generated videos, preview images, cache files, and job records? Source uploads will stay."
  );
  if (!confirmed) return;

  window.clearTimeout(state.previewTimer);
  state.previewDirty = false;
  els.clearGeneratedButton.disabled = true;
  setActivity("Cleaning generated files", null);
  try {
    const result = await api("/api/cleanup-generated", { method: "POST" });
    resetGeneratedUi();
    setActivity(
      `Cleaned all generated/cache data: ${result.filesDeleted} files and ${result.directoriesDeleted} folders (${formatBytes(result.bytesFreed)} freed). Source uploads preserved.`,
      0
    );
    await refreshStatus();
  } catch (error) {
    setActivity(`Cleanup failed: ${error.message}`, null);
  } finally {
    setExportEnabled(true);
  }
}

function clearFramePlayback() {
  if (state.framePlaybackTimer) {
    window.clearTimeout(state.framePlaybackTimer);
  }
  state.framePlaybackTimer = null;
  state.frameEventQueue = [];
  state.frameEventsAfter = 0;
  state.frameEventsInFlight = false;
  state.lastDisplayedFrameSeq = 0;
}

function displayFrameEvent(frame) {
  if (!frame?.originalUrl || !frame?.enhancedUrl) return;
  const cacheBust = `?job=${encodeURIComponent(state.currentJobId || "")}&frame=${frame.seq}`;
  const originalUrl = `${frame.originalUrl}${cacheBust}`;
  const enhancedUrl = `${frame.enhancedUrl}${cacheBust}`;
  if (state.liveOriginalUrl !== originalUrl) {
    els.originalPreview.src = originalUrl;
    state.liveOriginalUrl = originalUrl;
  }
  if (state.liveEnhancedUrl !== enhancedUrl) {
    els.enhancedPreview.src = enhancedUrl;
    state.liveEnhancedUrl = enhancedUrl;
  }

  els.sourceVideo.hidden = true;
  els.originalPreview.hidden = false;
  els.enhancedPreview.hidden = false;
  els.originalEmpty.hidden = true;
  els.enhancedEmpty.hidden = true;
  if (Number.isFinite(frame.seconds)) {
    els.timeline.value = String(frame.seconds);
    els.currentTime.textContent = formatTime(frame.seconds);
  }
  els.originalInfo.textContent = `frame ${frame.frameIndex} | ${formatTime(frame.seconds)}`;
  els.enhancedInfo.textContent = frame.cached
    ? `frame ${frame.frameIndex} / ${frame.framesTotal} | cached`
    : `frame ${frame.frameIndex} / ${frame.framesTotal} | generated`;
  state.lastDisplayedFrameSeq = frame.seq;
}

function framePlaybackDelay() {
  if (state.frameEventQueue.length > 120) return 30;
  if (state.frameEventQueue.length > 40) return 60;
  return 120;
}

function playFrameQueue() {
  if (state.framePlaybackTimer || !state.frameEventQueue.length) return;
  displayFrameEvent(state.frameEventQueue.shift());
  state.framePlaybackTimer = window.setTimeout(() => {
    state.framePlaybackTimer = null;
    playFrameQueue();
  }, framePlaybackDelay());
}

async function fetchFrameEvents(jobId, { allowInactive = false } = {}) {
  if (!jobId || state.frameEventsInFlight) return false;
  state.frameEventsInFlight = true;
  try {
    const result = await api(`/api/jobs/${jobId}/frames?after=${state.frameEventsAfter}&limit=240`);
    if (!allowInactive && jobId !== state.currentJobId) return false;
    if (result.frames?.length) {
      state.frameEventsAfter = result.nextAfter;
      state.frameEventQueue.push(...result.frames);
      playFrameQueue();
    }
    if (result.hasMore && !allowInactive && jobId === state.currentJobId) {
      window.setTimeout(() => fetchFrameEvents(jobId), 50);
    }
    return Boolean(result.hasMore);
  } catch (error) {
    if (jobId === state.currentJobId) {
      setActivity(`Frame viewer update failed: ${error.message}`, null);
    }
    return false;
  } finally {
    state.frameEventsInFlight = false;
  }
}

async function fetchRemainingFrameEvents(jobId) {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    const hasMore = await fetchFrameEvents(jobId, { allowInactive: true });
    if (!hasMore) return;
  }
}

function updateLiveFrame(job) {
  if (!job.currentOriginalUrl || !job.currentEnhancedUrl) return;
  const originalUrl = `${job.currentOriginalUrl}?t=${job.updatedAt}`;
  const enhancedUrl = `${job.currentEnhancedUrl}?t=${job.updatedAt}`;
  if (state.liveOriginalUrl !== originalUrl) {
    els.originalPreview.src = originalUrl;
    state.liveOriginalUrl = originalUrl;
  }
  if (state.liveEnhancedUrl !== enhancedUrl) {
    els.enhancedPreview.src = enhancedUrl;
    state.liveEnhancedUrl = enhancedUrl;
  }
  els.sourceVideo.hidden = true;
  els.originalPreview.hidden = false;
  els.enhancedPreview.hidden = false;
  els.originalEmpty.hidden = true;
  els.enhancedEmpty.hidden = true;
  if (Number.isFinite(job.currentFrameSeconds)) {
    els.timeline.value = String(job.currentFrameSeconds);
    els.currentTime.textContent = formatTime(job.currentFrameSeconds);
  }
  els.originalInfo.textContent = job.currentFrameIndex
    ? `frame ${job.currentFrameIndex}`
    : "source frame";
  els.enhancedInfo.textContent = job.framesTotal
    ? `frame ${job.currentFrameIndex} / ${job.framesTotal}`
    : "enhanced frame";
}

function updateJobUi(job) {
  setActivity(job.message, job.progress);
  if (job.framesTotal) {
    const cached = job.cacheHits ? ` | cached ${job.cacheHits}` : "";
    els.frameProgressText.textContent = `Frames: ${job.framesDone} / ${job.framesTotal}${cached}`;
  } else {
    els.frameProgressText.textContent = "Frames: preparing";
  }
  els.etaText.textContent = `ETA: ${formatDuration(job.etaSeconds)}`;
  if (state.frameEventsAfter === 0 && !state.frameEventQueue.length) {
    updateLiveFrame(job);
  }
}

function pollJob(jobId) {
  if (state.jobTimer) clearInterval(state.jobTimer);
  state.jobTimer = setInterval(async () => {
    try {
      const job = await api(`/api/jobs/${jobId}`);
      await fetchFrameEvents(jobId);
      updateJobUi(job);
      if (job.status === "done") {
        await fetchRemainingFrameEvents(jobId);
        clearInterval(state.jobTimer);
        state.jobTimer = null;
        els.downloadLink.href = job.outputUrl;
        els.downloadLink.hidden = false;
        state.exportInFlight = false;
        state.currentJobId = null;
        saveLocalState();
        setExportEnabled(true);
        await refreshStatus();
      }
      if (job.status === "error" || job.status === "cancelled") {
        clearInterval(state.jobTimer);
        state.jobTimer = null;
        setActivity(job.status === "cancelled" ? job.message : `Export failed: ${job.error || job.message}`, job.progress);
        state.exportInFlight = false;
        state.currentJobId = null;
        saveLocalState();
        setExportEnabled(true);
      }
    } catch (error) {
      clearInterval(state.jobTimer);
      state.jobTimer = null;
      setActivity(`Job polling failed: ${error.message}`, 0);
      state.exportInFlight = false;
      state.currentJobId = null;
      saveLocalState();
      setExportEnabled(true);
    }
  }, 700);
}

els.videoInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    await uploadVideo(file);
  } catch (error) {
    setActivity(`Upload failed: ${error.message}`, 0);
  }
});

els.timeline.addEventListener("input", () => {
  const seconds = Number(els.timeline.value);
  els.currentTime.textContent = formatTime(seconds);
  if (state.video) {
    els.sourceVideo.currentTime = seconds;
    scheduleAutoPreview({ delay: 500, reason: "playhead moved" });
  }
});

els.sourceVideo.addEventListener("timeupdate", () => {
  if (document.activeElement === els.timeline) return;
  const seconds = els.sourceVideo.currentTime;
  if (Number.isFinite(seconds)) {
    els.timeline.value = String(seconds);
    els.currentTime.textContent = formatTime(seconds);
  }
});

els.sourceVideo.addEventListener("seeked", () => {
  if (!state.video || document.activeElement === els.timeline) return;
  const seconds = els.sourceVideo.currentTime;
  if (Number.isFinite(seconds)) {
    els.timeline.value = String(seconds);
    els.currentTime.textContent = formatTime(seconds);
    scheduleAutoPreview({ delay: 300, reason: "playhead moved" });
  }
});

els.exportButton.addEventListener("click", startExport);
els.cancelExportButton.addEventListener("click", cancelExport);
els.openOutputFolderButton.addEventListener("click", openOutputFolder);
els.clearGeneratedButton.addEventListener("click", clearGeneratedFiles);
els.refreshStatusButton.addEventListener("click", refreshStatus);
els.scaleBy.addEventListener("change", () => {
  updateScaleMode();
  saveLocalState();
  scheduleAutoPreview();
});
els.upscale.addEventListener("change", () => {
  saveLocalState();
  scheduleAutoPreview();
});
els.targetLongestSide.addEventListener("input", () => {
  saveLocalState();
  scheduleAutoPreview();
});
els.patchSize.addEventListener("change", () => {
  collectSettings();
  saveLocalState();
  scheduleAutoPreview();
});
els.stride.addEventListener("change", () => {
  collectSettings();
  saveLocalState();
  scheduleAutoPreview();
});
els.temporalConsistency.addEventListener("change", saveLocalState);
els.seed.addEventListener("input", () => {
  saveLocalState();
  scheduleAutoPreview();
});
els.prompt.addEventListener("input", () => {
  saveLocalState();
  scheduleAutoPreview();
});
els.encoder.addEventListener("change", saveLocalState);
els.crf.addEventListener("input", () => {
  els.crfValue.textContent = els.crf.value;
  saveLocalState();
});

updateScaleMode();
setupTooltips();
refreshStatus();
restoreSession();
