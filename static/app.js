const state = {
  video: null,
  jobTimer: null,
  previewTimer: null,
  previewInFlight: false,
  previewDirty: false,
  previewVersion: 0,
  exportInFlight: false,
};

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
  activityText: document.querySelector("#activityText"),
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
  seed: document.querySelector("#seed"),
  prompt: document.querySelector("#prompt"),
  encoder: document.querySelector("#encoder"),
  crf: document.querySelector("#crf"),
  crfValue: document.querySelector("#crfValue"),
};

function formatTime(seconds) {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const minutes = Math.floor(safe / 60);
  const secs = Math.floor(safe % 60);
  const ms = Math.floor((safe - Math.floor(safe)) * 1000);
  return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}.${String(ms).padStart(3, "0")}`;
}

function setActivity(message, progress = null) {
  els.activityText.textContent = message;
  if (progress !== null) {
    els.jobProgress.style.width = `${Math.max(0, Math.min(1, progress)) * 100}%`;
  }
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

function setVideo(record) {
  state.video = record;
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
  els.exportButton.disabled = false;
  els.downloadLink.hidden = true;
  setActivity("Preparing preview", 0.1);
  scheduleAutoPreview({ delay: 200, reason: "new video" });
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
  state.exportInFlight = true;
  state.previewDirty = false;
  setExportEnabled(false);
  els.downloadLink.hidden = true;
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
    pollJob(job.id);
  } catch (error) {
    setActivity(`Export failed: ${error.message}`, 0);
    state.exportInFlight = false;
    setExportEnabled(true);
  }
}

function pollJob(jobId) {
  if (state.jobTimer) clearInterval(state.jobTimer);
  state.jobTimer = setInterval(async () => {
    try {
      const job = await api(`/api/jobs/${jobId}`);
      setActivity(job.message, job.progress);
      if (job.status === "done") {
        clearInterval(state.jobTimer);
        state.jobTimer = null;
        els.downloadLink.href = job.outputUrl;
        els.downloadLink.hidden = false;
        state.exportInFlight = false;
        setExportEnabled(true);
        await refreshStatus();
      }
      if (job.status === "error") {
        clearInterval(state.jobTimer);
        state.jobTimer = null;
        setActivity(`Export failed: ${job.error || job.message}`, 0);
        state.exportInFlight = false;
        setExportEnabled(true);
      }
    } catch (error) {
      clearInterval(state.jobTimer);
      state.jobTimer = null;
      setActivity(`Job polling failed: ${error.message}`, 0);
      state.exportInFlight = false;
      setExportEnabled(true);
    }
  }, 1600);
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
els.refreshStatusButton.addEventListener("click", refreshStatus);
els.scaleBy.addEventListener("change", () => {
  updateScaleMode();
  scheduleAutoPreview();
});
els.upscale.addEventListener("change", () => scheduleAutoPreview());
els.targetLongestSide.addEventListener("input", () => scheduleAutoPreview());
els.patchSize.addEventListener("change", () => {
  collectSettings();
  scheduleAutoPreview();
});
els.stride.addEventListener("change", () => {
  collectSettings();
  scheduleAutoPreview();
});
els.seed.addEventListener("input", () => scheduleAutoPreview());
els.prompt.addEventListener("input", () => scheduleAutoPreview());
els.crf.addEventListener("input", () => {
  els.crfValue.textContent = els.crf.value;
});

updateScaleMode();
refreshStatus();
