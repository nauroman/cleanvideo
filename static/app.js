const defaultEngine = "flashvsr";
const settingsStorageVersion = 2;
const settingsStorageKey = "cleanvideo.session.v1";
const engineMetricsStorageKey = "cleanvideo.engineMetrics.v1";
const previewTimeoutMsByEngine = {
  flashvsr: 5 * 60 * 1000,
  seedvr2: 3 * 60 * 1000,
  hypir: 2 * 60 * 1000,
};

const state = {
  engine: defaultEngine,
  video: null,
  jobTimer: null,
  currentJobId: null,
  previewTimer: null,
  previewController: null,
  previewInFlight: false,
  previewDirty: false,
  previewVersion: 0,
  exportInFlight: false,
  partialExportInFlight: false,
  adapterInFlight: false,
  currentAdapterJobId: null,
  adapterTimer: null,
  adapters: [],
  adaptersById: new Map(),
  liveOriginalUrl: null,
  liveEnhancedUrl: null,
  frameEventQueue: [],
  readyFrameEvents: [],
  readyFrameSeqs: new Set(),
  frameEventsAfter: 0,
  frameEventsInFlight: false,
  framePlaybackTimer: null,
  readyPlaybackTimer: null,
  readyPlaybackActive: false,
  readyPlaybackIndex: 0,
  lastDisplayedFrameSeq: 0,
  lastExportFramesDone: 0,
  lastExportFramesTotal: 0,
  jobPollFailures: 0,
  adapterPollFailures: 0,
  status: null,
  engineMetrics: {},
};

const resolutionPresets = {
  preset_720p: 1280,
  preset_fullhd: 1920,
  preset_4k: 3840,
  preset_8k: 7680,
};

const engineLabels = {
  hypir: "HYPIR",
  seedvr2: "SeedVR2",
  flashvsr: "FlashVSR",
};

const els = {
  engineCards: Array.from(document.querySelectorAll(".engine-card[data-engine]")),
  videoInput: document.querySelector("#videoInput"),
  videoName: document.querySelector("#videoName"),
  videoMeta: document.querySelector("#videoMeta"),
  comparisonBox: document.querySelector("#comparisonBox"),
  comparisonAfterClip: document.querySelector("#comparisonAfterClip"),
  comparisonDivider: document.querySelector("#comparisonDivider"),
  comparisonHandle: document.querySelector("#comparisonHandle"),
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
  previewButton: document.querySelector("#previewButton"),
  playReadyButton: document.querySelector("#playReadyButton"),
  playReadyLabel: document.querySelector("#playReadyLabel"),
  exportButton: document.querySelector("#exportButton"),
  cancelExportButton: document.querySelector("#cancelExportButton"),
  partialExportButton: document.querySelector("#partialExportButton"),
  openOutputFolderButton: document.querySelector("#openOutputFolderButton"),
  clearGeneratedButton: document.querySelector("#clearGeneratedButton"),
  activityText: document.querySelector("#activityText"),
  frameProgressText: document.querySelector("#frameProgressText"),
  etaText: document.querySelector("#etaText"),
  jobProgress: document.querySelector("#jobProgress"),
  downloadLink: document.querySelector("#downloadLink"),
  partialDownloadLink: document.querySelector("#partialDownloadLink"),
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
  adapterSelect: document.querySelector("#adapterSelect"),
  secondPass: document.querySelector("#secondPass"),
  adapterQuality: document.querySelector("#adapterQuality"),
  seedvr2BatchSize: document.querySelector("#seedvr2BatchSize"),
  seedvr2TemporalOverlap: document.querySelector("#seedvr2TemporalOverlap"),
  seedvr2ChunkSize: document.querySelector("#seedvr2ChunkSize"),
  seedvr2ColorCorrection: document.querySelector("#seedvr2ColorCorrection"),
  seedvr2PreviewSize: document.querySelector("#seedvr2PreviewSize"),
  flashvsrVariant: document.querySelector("#flashvsrVariant"),
  flashvsrSparseRatio: document.querySelector("#flashvsrSparseRatio"),
  flashvsrLocalRange: document.querySelector("#flashvsrLocalRange"),
  flashvsrPreviewCap: document.querySelector("#flashvsrPreviewCap"),
  deleteAdapterButton: document.querySelector("#deleteAdapterButton"),
  deleteAllAdaptersButton: document.querySelector("#deleteAllAdaptersButton"),
  trainAdapterButton: document.querySelector("#trainAdapterButton"),
  cancelAdapterButton: document.querySelector("#cancelAdapterButton"),
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

let comparisonDragging = false;

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

function compactUiMessage(message, limit = 260) {
  const raw = String(message || "");
  const singleLine = raw.replace(/\s+/g, " ").trim();
  if (singleLine.length <= limit) return singleLine;
  return `${singleLine.slice(0, Math.max(0, limit - 1)).trim()}…`;
}

function setActivity(message, progress = null) {
  const raw = String(message || "");
  els.activityText.textContent = compactUiMessage(raw);
  els.activityText.title = raw;
  if (progress !== null) {
    els.jobProgress.style.width = `${Math.max(0, Math.min(1, progress)) * 100}%`;
  }
}

function updateComparisonVisibility() {
  const active = !els.originalPreview.hidden && !els.enhancedPreview.hidden;
  els.comparisonAfterClip.hidden = !active;
  els.comparisonDivider.hidden = !active;
  els.comparisonHandle.hidden = !active;
  els.comparisonBox.classList.toggle("comparison-active", active);
}

function setComparisonPositionFromClientX(clientX) {
  const rect = els.comparisonBox.getBoundingClientRect();
  if (!rect.width) return;
  const percent = Math.max(5, Math.min(95, ((clientX - rect.left) / rect.width) * 100));
  els.comparisonBox.style.setProperty("--compare-pos", `${percent.toFixed(2)}%`);
}

function comparisonPosition() {
  const raw = getComputedStyle(els.comparisonBox).getPropertyValue("--compare-pos").trim() || "50%";
  const numeric = Number.parseFloat(raw);
  return Number.isFinite(numeric) ? numeric : 50;
}

function setComparisonPosition(percent) {
  const clamped = Math.max(5, Math.min(95, percent));
  els.comparisonBox.style.setProperty("--compare-pos", `${clamped.toFixed(2)}%`);
}

function beginComparisonDrag(event) {
  if (!els.comparisonBox.classList.contains("comparison-active")) return;
  event.preventDefault();
  comparisonDragging = true;
  setComparisonPositionFromClientX(event.clientX);
}

function moveComparisonDrag(event) {
  if (!comparisonDragging) return;
  event.preventDefault();
  setComparisonPositionFromClientX(event.clientX);
}

function endComparisonDrag() {
  comparisonDragging = false;
}

function showSourceVideo() {
  els.sourceVideo.hidden = false;
  els.originalPreview.hidden = true;
  els.enhancedPreview.hidden = true;
  els.originalEmpty.hidden = true;
  els.enhancedEmpty.hidden = true;
  updateComparisonVisibility();
}

function showFramePair({ originalUrl, enhancedUrl, originalInfo, enhancedInfo }) {
  if (originalUrl && state.liveOriginalUrl !== originalUrl) {
    els.originalPreview.src = originalUrl;
    state.liveOriginalUrl = originalUrl;
  }
  if (enhancedUrl && state.liveEnhancedUrl !== enhancedUrl) {
    els.enhancedPreview.src = enhancedUrl;
    state.liveEnhancedUrl = enhancedUrl;
  }
  els.sourceVideo.hidden = true;
  els.originalPreview.hidden = false;
  els.enhancedPreview.hidden = false;
  els.originalEmpty.hidden = true;
  els.enhancedEmpty.hidden = true;
  if (originalInfo !== undefined) {
    els.originalInfo.textContent = originalInfo;
  }
  if (enhancedInfo !== undefined) {
    els.enhancedInfo.textContent = enhancedInfo;
  }
  updateComparisonVisibility();
}

function stopReadyPlayback() {
  if (state.readyPlaybackTimer) {
    window.clearTimeout(state.readyPlaybackTimer);
  }
  state.readyPlaybackTimer = null;
  state.readyPlaybackActive = false;
  state.readyPlaybackIndex = 0;
  updateReadyPlaybackButton();
}

function updateReadyPlaybackButton() {
  const count = state.readyFrameEvents.length;
  els.playReadyButton.hidden = !state.exportInFlight && count === 0;
  els.playReadyButton.disabled = count === 0;
  els.playReadyLabel.textContent = state.readyPlaybackActive ? "Stop Ready" : `Play Ready${count ? ` (${count})` : ""}`;
  els.playReadyButton.title = count
    ? "Play generated PNG frames at the source FPS"
    : "No generated PNG frames are ready yet";
}

function rememberReadyFrames(frames = []) {
  for (const frame of frames) {
    if (!frame?.originalUrl || !frame?.enhancedUrl || state.readyFrameSeqs.has(frame.seq)) continue;
    state.readyFrameSeqs.add(frame.seq);
    state.readyFrameEvents.push(frame);
  }
  state.readyFrameEvents.sort((a, b) => (a.frameIndex || a.seq) - (b.frameIndex || b.seq));
  updateReadyPlaybackButton();
}

function readyPlaybackDelay() {
  const fps = Number(state.video?.metadata?.fps || 30);
  return Math.max(8, Math.round(1000 / Math.max(1, fps)));
}

function tickReadyPlayback() {
  if (!state.readyPlaybackActive) return;
  const frames = state.readyFrameEvents;
  if (!frames.length || state.readyPlaybackIndex >= frames.length) {
    stopReadyPlayback();
    return;
  }
  displayFrameEvent(frames[state.readyPlaybackIndex]);
  state.readyPlaybackIndex += 1;
  state.readyPlaybackTimer = window.setTimeout(tickReadyPlayback, readyPlaybackDelay());
}

function toggleReadyPlayback() {
  if (state.readyPlaybackActive) {
    stopReadyPlayback();
    return;
  }
  if (!state.readyFrameEvents.length) return;
  if (state.framePlaybackTimer) {
    window.clearTimeout(state.framePlaybackTimer);
    state.framePlaybackTimer = null;
  }
  state.readyPlaybackActive = true;
  state.readyPlaybackIndex = 0;
  updateReadyPlaybackButton();
  tickReadyPlayback();
}

function cancelActivePreviewRequest() {
  window.clearTimeout(state.previewTimer);
  state.previewDirty = false;
  state.previewController?.abort();
  if (state.previewInFlight) {
    fetch("/api/preview/cancel", { method: "POST" }).catch(() => {});
  }
  state.previewVersion += 1;
  state.previewInFlight = false;
  state.previewController = null;
  setExportEnabled(true);
}

function collectUiSettings() {
  return {
    engine: state.engine,
    scaleBy: els.scaleBy.value,
    upscale: els.upscale.value,
    targetLongestSide: els.targetLongestSide.value,
    patchSize: els.patchSize.value,
    stride: els.stride.value,
    temporalConsistency: els.temporalConsistency.value,
    adapterId: els.adapterSelect.value,
    secondPass: els.secondPass.value,
    adapterQuality: els.adapterQuality.value,
    seedvr2BatchSize: els.seedvr2BatchSize.value,
    seedvr2TemporalOverlap: els.seedvr2TemporalOverlap.value,
    seedvr2ChunkSize: els.seedvr2ChunkSize.value,
    seedvr2ColorCorrection: els.seedvr2ColorCorrection.value,
    seedvr2PreviewSize: els.seedvr2PreviewSize.value,
    flashvsrVariant: els.flashvsrVariant.value,
    flashvsrSparseRatio: els.flashvsrSparseRatio.value,
    flashvsrLocalRange: els.flashvsrLocalRange.value,
    flashvsrPreviewCap: els.flashvsrPreviewCap.value,
    seed: els.seed.value,
    prompt: els.prompt.value,
    encoder: els.encoder.value,
    crf: els.crf.value,
  };
}

function saveLocalState() {
  const payload = {
    version: settingsStorageVersion,
    settings: collectUiSettings(),
    videoId: state.video?.id ?? null,
    currentJobId: state.exportInFlight ? state.currentJobId : null,
    currentAdapterJobId: state.adapterInFlight ? state.currentAdapterJobId : null,
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

function readEngineMetrics() {
  try {
    return JSON.parse(localStorage.getItem(engineMetricsStorageKey) || "{}");
  } catch {
    return {};
  }
}

function saveEngineMetrics() {
  localStorage.setItem(engineMetricsStorageKey, JSON.stringify(state.engineMetrics));
}

function formatSecondsPerFrame(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return "--";
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms/frame`;
  if (seconds < 10) return `${seconds.toFixed(1)}s/frame`;
  return `${Math.round(seconds)}s/frame`;
}

function metricLabel(metric) {
  if (!metric?.averageFrameSeconds) return "Speed: --";
  const source = metric.source === "export" ? "Export" : "Preview";
  const resolution = metric.resolution ? ` @ ${metric.resolution}` : "";
  return `${source}: ${formatSecondsPerFrame(metric.averageFrameSeconds)}${resolution}`;
}

function updateEngineSpeedLabels() {
  const videoMetrics = state.video ? state.engineMetrics[state.video.id] || {} : {};
  document.querySelectorAll("[data-engine-speed]").forEach((element) => {
    element.textContent = metricLabel(videoMetrics[element.dataset.engineSpeed]);
  });
}

function rememberEngineMetric(engine, metric) {
  if (!state.video || !engine || !metric?.averageFrameSeconds) return;
  const videoMetrics = state.engineMetrics[state.video.id] || {};
  videoMetrics[engine] = {
    averageFrameSeconds: Number(metric.averageFrameSeconds),
    resolution: metric.resolution || null,
    source: metric.source || "preview",
    updatedAt: new Date().toISOString(),
  };
  state.engineMetrics[state.video.id] = videoMetrics;
  saveEngineMetrics();
  updateEngineSpeedLabels();
}

function applySavedSettings(settings = {}, { restoreEngine = true } = {}) {
  if (restoreEngine && settings.engine && engineLabels[settings.engine]) {
    state.engine = settings.engine;
  }
  const entries = [
    ["scaleBy", els.scaleBy],
    ["upscale", els.upscale],
    ["targetLongestSide", els.targetLongestSide],
    ["patchSize", els.patchSize],
    ["stride", els.stride],
    ["temporalConsistency", els.temporalConsistency],
    ["adapterId", els.adapterSelect],
    ["secondPass", els.secondPass],
    ["adapterQuality", els.adapterQuality],
    ["seedvr2BatchSize", els.seedvr2BatchSize],
    ["seedvr2TemporalOverlap", els.seedvr2TemporalOverlap],
    ["seedvr2ChunkSize", els.seedvr2ChunkSize],
    ["seedvr2ColorCorrection", els.seedvr2ColorCorrection],
    ["seedvr2PreviewSize", els.seedvr2PreviewSize],
    ["flashvsrVariant", els.flashvsrVariant],
    ["flashvsrSparseRatio", els.flashvsrSparseRatio],
    ["flashvsrLocalRange", els.flashvsrLocalRange],
    ["flashvsrPreviewCap", els.flashvsrPreviewCap],
    ["seed", els.seed],
    ["prompt", els.prompt],
    ["encoder", els.encoder],
    ["crf", els.crf],
  ];
  for (const [key, element] of entries) {
    if (settings[key] !== undefined && element) {
      if (element.tagName === "SELECT") {
        const hasOption = Array.from(element.options).some((option) => option.value === String(settings[key]));
        if (!hasOption) continue;
      }
      element.value = settings[key];
    }
  }
  els.crfValue.textContent = els.crf.value;
  updateScaleMode();
  updateEngineUi();
  updateDeleteAdapterButton();
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
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

async function loadAdapters(selectedId = els.adapterSelect.value || "base") {
  const result = await api("/api/adapters");
  const ids = new Set();
  state.adapters = result.adapters || [];
  state.adaptersById = new Map(state.adapters.map((adapter) => [adapter.id, adapter]));
  els.adapterSelect.replaceChildren();
  for (const adapter of state.adapters) {
    ids.add(adapter.id);
    const option = document.createElement("option");
    option.value = adapter.id;
    option.textContent = adapter.name || adapter.id;
    els.adapterSelect.appendChild(option);
  }
  const rootSelectedId = String(selectedId).split("@step-", 1)[0];
  els.adapterSelect.value = ids.has(selectedId) ? selectedId : ids.has(rootSelectedId) ? rootSelectedId : "base";
  updateDeleteAdapterButton();
}

function selectedAdapterRecord() {
  return state.adaptersById.get(els.adapterSelect.value) || null;
}

function selectedRootAdapterRecord() {
  const selected = selectedAdapterRecord();
  if (!selected || selected.id === "base") return null;
  const rootId = selected.parentAdapterId || selected.id.split("@step-")[0];
  return state.adaptersById.get(rootId) || selected;
}

function canDeleteSelectedAdapter() {
  return Boolean(
    state.engine === "hypir"
    && selectedRootAdapterRecord()
    && !state.exportInFlight
    && !state.previewInFlight
    && !state.adapterInFlight
  );
}

function rootAdapterRecords() {
  const roots = new Map();
  for (const adapter of state.adapters) {
    if (!adapter || adapter.id === "base" || adapter.parentAdapterId) continue;
    roots.set(adapter.id, adapter);
  }
  return Array.from(roots.values());
}

function canDeleteAllAdapters() {
  return state.engine === "hypir"
    && rootAdapterRecords().length > 0
    && !state.exportInFlight
    && !state.previewInFlight
    && !state.adapterInFlight;
}

function updateDeleteAdapterButton() {
  const rootAdapter = selectedRootAdapterRecord();
  els.deleteAdapterButton.disabled = !canDeleteSelectedAdapter();
  els.deleteAdapterButton.title = rootAdapter
    ? `Delete ${rootAdapter.name || rootAdapter.id}`
    : "Base HYPIR cannot be deleted";
  els.deleteAllAdaptersButton.disabled = !canDeleteAllAdapters();
  els.deleteAllAdaptersButton.title = rootAdapterRecords().length
    ? "Delete all film adapters"
    : "No film adapters to delete";
}

function engineStatus(engine = state.engine) {
  return state.status?.engines?.[engine] || state.status?.[engine] || null;
}

function isEngineAvailable(engine = state.engine) {
  const status = engineStatus(engine);
  if (!status) return engine === "hypir";
  if (engine === "hypir") {
    return Boolean(status.cudaAvailable && status.weightPresent && status.baseModelPresent);
  }
  return Boolean(status.available);
}

function supportsPartialExport(engine = state.engine) {
  return engine === "hypir" || engine === "flashvsr";
}

function enhancedInfoLabel(engine = state.engine) {
  if (engine === "seedvr2" || engine === "flashvsr") {
    return `${engineLabels[engine] || engine} auto preview`;
  }
  return `${engineLabels[engine] || engine} preview`;
}

function updateEngineSpecificControls() {
  document.querySelectorAll("[data-engine-field]").forEach((element) => {
    const engines = String(element.dataset.engineField || "").split(/\s+/).filter(Boolean);
    const visible = engines.includes(state.engine);
    element.hidden = !visible;
  });
}

function updateEngineUi() {
  updateEngineSpecificControls();
  updateEngineSpeedLabels();
  for (const card of els.engineCards) {
    const engine = card.dataset.engine;
    const selected = engine === state.engine;
    const status = engineStatus(engine);
    card.classList.toggle("selected", selected);
    card.classList.toggle("blocked", Boolean(status && !isEngineAvailable(engine)));
    card.setAttribute("aria-pressed", selected ? "true" : "false");
    const small = card.querySelector("small");
    if (!small) continue;
    if (engine === "hypir") {
      small.textContent = status?.loaded ? "Loaded" : "SD2 local CUDA";
    } else if (status && !isEngineAvailable(engine)) {
      small.textContent = "Blocked";
    } else if (status?.available) {
      small.textContent = "Ready";
    }
  }
  const selectedLabel = engineLabels[state.engine] || state.engine;
  const unavailable = !isEngineAvailable();
  const status = engineStatus();
  els.enhancedInfo.textContent = enhancedInfoLabel();
  if (status) {
    if (unavailable) {
      els.modelStatus.textContent = "blocked";
    } else if (status.loaded) {
      els.modelStatus.textContent = "loaded";
    } else if (status.available) {
      els.modelStatus.textContent = "ready";
    } else {
      els.modelStatus.textContent = "cold";
    }
  }
  els.trainAdapterButton.title = state.engine === "hypir"
    ? "Create film-specific HYPIR adapter"
    : "Film adapters are currently available only for HYPIR";
  els.previewButton.hidden = true;
  els.previewButton.title = `${selectedLabel} previews run automatically from the playhead`;
  if (!supportsPartialExport()) {
    els.partialExportButton.title = "Partial export is available for HYPIR frames and completed FlashVSR chunks";
  }
  if (unavailable && status) {
    const reason = status?.blockedReason || status?.missing?.join(", ") || "engine is not ready";
    setActivity(`${selectedLabel} blocked: ${reason}`, 0);
  } else if (!unavailable && els.activityText.textContent.includes(" blocked:")) {
    setActivity("Idle", 0);
  }
}

function setEngine(engine, { persist = true, preview = true } = {}) {
  if (!engineLabels[engine] || state.engine === engine) return;
  if (state.previewInFlight || state.previewTimer) {
    cancelActivePreviewRequest();
  }
  state.engine = engine;
  if (engine !== "hypir") {
    els.adapterSelect.value = "base";
    els.secondPass.value = "off";
  }
  updateEngineUi();
  updateDeleteAdapterButton();
  setExportEnabled(!state.exportInFlight && !state.previewInFlight && !state.adapterInFlight);
  if (persist) saveLocalState();
  if (preview && state.video) {
    scheduleAutoPreview({ delay: 250, reason: `${engineLabels[engine]} selected` });
  }
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
    engine: state.engine,
    prompt: els.prompt.value,
    scaleBy,
    upscale: Number(els.upscale.value),
    targetLongestSide: scaleBy === "longest_side" ? presetLongestSide || Number(els.targetLongestSide.value) : null,
    patchSize,
    stride,
    temporalConsistency: els.temporalConsistency.value,
    adapterId: state.engine === "hypir" ? els.adapterSelect.value : "base",
    secondPass: state.engine === "hypir" ? els.secondPass.value : "off",
    seedvr2BatchSize: Number(els.seedvr2BatchSize.value),
    seedvr2TemporalOverlap: Number(els.seedvr2TemporalOverlap.value),
    seedvr2ChunkSize: Number(els.seedvr2ChunkSize.value),
    seedvr2ColorCorrection: els.seedvr2ColorCorrection.value,
    seedvr2PreviewSize: Number(els.seedvr2PreviewSize.value),
    flashvsrVariant: els.flashvsrVariant.value,
    flashvsrSparseRatio: Number(els.flashvsrSparseRatio.value),
    flashvsrLocalRange: Number(els.flashvsrLocalRange.value),
    flashvsrPreviewCap: Number(els.flashvsrPreviewCap.value),
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
  stopReadyPlayback();
  state.readyFrameEvents = [];
  state.readyFrameSeqs = new Set();
  state.currentJobId = null;
  state.partialExportInFlight = false;
  state.lastExportFramesDone = 0;
  state.lastExportFramesTotal = 0;
  state.liveOriginalUrl = null;
  state.liveEnhancedUrl = null;
  els.downloadLink.hidden = true;
  els.downloadLink.removeAttribute("href");
  els.partialDownloadLink.hidden = true;
  els.partialDownloadLink.removeAttribute("href");
  els.originalPreview.hidden = true;
  els.originalPreview.removeAttribute("src");
  els.enhancedPreview.hidden = true;
  els.enhancedPreview.removeAttribute("src");
  if (state.video) {
    showSourceVideo();
    els.originalEmpty.hidden = true;
  } else {
    els.sourceVideo.hidden = true;
    els.originalEmpty.hidden = false;
    els.enhancedEmpty.hidden = true;
    updateComparisonVisibility();
  }
  els.originalInfo.textContent = "source frame";
  els.enhancedInfo.textContent = enhancedInfoLabel();
  els.frameProgressText.textContent = "Frames: --";
  els.etaText.textContent = "ETA: --";
  els.jobProgress.style.width = "0%";
  updatePartialExportButton();
  updateReadyPlaybackButton();
}

function setVideo(record, { persist = true } = {}) {
  state.video = record;
  state.liveOriginalUrl = null;
  state.liveEnhancedUrl = null;
  const meta = record.metadata;
  els.videoName.textContent = record.name;
  els.videoMeta.textContent = `${meta.width}x${meta.height} | ${meta.fps.toFixed(3)} fps | ${formatTime(meta.duration)} | ${meta.codec || "video"}`;
  els.sourceVideo.src = record.url;
  showSourceVideo();
  els.originalPreview.hidden = true;
  els.enhancedPreview.hidden = true;
  updateComparisonVisibility();
  els.timeline.max = String(meta.duration || 0);
  els.timeline.value = "0";
  els.currentTime.textContent = formatTime(0);
  els.durationTime.textContent = formatTime(meta.duration || 0);
  els.frameProgressText.textContent = "Frames: --";
  els.etaText.textContent = "ETA: --";
  setExportEnabled(true);
  els.downloadLink.hidden = true;
  els.partialDownloadLink.hidden = true;
  els.partialDownloadLink.removeAttribute("href");
  state.partialExportInFlight = false;
  state.lastExportFramesDone = 0;
  state.lastExportFramesTotal = 0;
  stopReadyPlayback();
  state.readyFrameEvents = [];
  state.readyFrameSeqs = new Set();
  updateReadyPlaybackButton();
  updatePartialExportButton();
  updateEngineSpeedLabels();
  if (persist) saveLocalState();
  setActivity("Preparing preview", 0.1);
  scheduleAutoPreview({ delay: 200, reason: "new video" });
}

async function restoreSession() {
  const saved = readLocalState();
  try {
    await loadAdapters(saved.settings?.adapterId || "base");
  } catch (error) {
    setActivity(`Could not load film adapters: ${error.message}`, 0);
  }
  applySavedSettings(saved.settings, { restoreEngine: saved.version === settingsStorageVersion });
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
    await restoreActiveAdapter(saved.currentAdapterJobId);
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
  state.jobPollFailures = 0;
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

async function restoreActiveAdapter(savedJobId) {
  let job = null;
  if (savedJobId) {
    try {
      job = await api(`/api/jobs/${savedJobId}`);
    } catch {
      job = null;
    }
  }
  if (!job || job.kind !== "adapter" || !["queued", "running"].includes(job.status)) {
    try {
      const result = await api("/api/jobs");
      job = result.jobs.find((candidate) => (
        candidate.kind === "adapter" && ["queued", "running"].includes(candidate.status)
      ));
    } catch {
      job = null;
    }
  }
  if (job) {
    resumeAdapterJob(job);
  }
}

function resumeAdapterJob(job) {
  state.adapterInFlight = true;
  state.currentAdapterJobId = job.id;
  state.adapterPollFailures = 0;
  updateAdapterJobUi(job);
  setExportEnabled(false);
  saveLocalState();
  pollAdapterJob(job.id);
}

async function refreshStatus() {
  try {
    const status = await api("/api/status");
    state.status = status;
    const selectedStatus = engineStatus();
    const gpu = selectedStatus?.gpu || status.hypir.gpu || status.flashvsr?.gpu || "no cuda";
    els.gpuStatus.textContent = (selectedStatus?.cudaAvailable ?? status.hypir.cudaAvailable)
      ? gpu.replace("NVIDIA GeForce ", "")
      : "none";
    els.nvencStatus.textContent = status.nvenc ? "ready" : "missing";
    if (!isEngineAvailable()) {
      els.modelStatus.textContent = "blocked";
    } else if (selectedStatus?.loaded) {
      els.modelStatus.textContent = "loaded";
    } else if (selectedStatus?.available) {
      els.modelStatus.textContent = "ready";
    } else {
      els.modelStatus.textContent = "cold";
    }
    updateEngineUi();
    setExportEnabled(!state.exportInFlight && !state.previewInFlight && !state.adapterInFlight);
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
  const engineReady = isEngineAvailable();
  const hypirSelected = state.engine === "hypir";
  els.previewButton.hidden = true;
  els.previewButton.disabled = true;
  els.exportButton.disabled = !enabled || !state.video || state.exportInFlight || state.previewInFlight || !engineReady;
  els.cancelExportButton.hidden = !state.exportInFlight;
  els.cancelExportButton.disabled = !state.exportInFlight;
  els.trainAdapterButton.disabled = !enabled || !state.video || state.exportInFlight || state.previewInFlight || state.adapterInFlight || !hypirSelected;
  els.cancelAdapterButton.hidden = !state.adapterInFlight;
  els.cancelAdapterButton.disabled = !state.adapterInFlight;
  els.adapterSelect.disabled = !hypirSelected || state.exportInFlight || state.previewInFlight || state.adapterInFlight;
  els.secondPass.disabled = !hypirSelected || state.exportInFlight || state.previewInFlight || state.adapterInFlight;
  els.adapterQuality.disabled = !hypirSelected || state.exportInFlight || state.previewInFlight || state.adapterInFlight;
  els.clearGeneratedButton.disabled = (
    state.exportInFlight
    || state.previewInFlight
    || state.adapterInFlight
    || state.partialExportInFlight
  );
  updatePartialExportButton();
  updateDeleteAdapterButton();
  updateEngineUi();
  updateReadyPlaybackButton();
}

function updatePartialExportButton() {
  const readyFrames = state.lastExportFramesDone;
  const totalFrames = state.lastExportFramesTotal;
  const supported = supportsPartialExport();
  const canSave = Boolean(
    state.exportInFlight
    && state.currentJobId
    && readyFrames > 0
    && !state.partialExportInFlight
    && supported
  );
  els.partialExportButton.hidden = !state.exportInFlight || !supported;
  els.partialExportButton.disabled = !canSave;
  if (state.partialExportInFlight) {
    els.partialExportButton.title = "Saving partial video";
  } else if (readyFrames > 0) {
    const totalText = totalFrames ? ` / ${totalFrames}` : "";
    const sourceText = state.engine === "flashvsr" ? "completed FlashVSR chunk frames" : "ready enhanced frames";
    els.partialExportButton.title = `Save ${readyFrames}${totalText} ${sourceText}`;
  } else {
    els.partialExportButton.title = state.engine === "flashvsr"
      ? "No completed FlashVSR chunks are ready yet"
      : "No enhanced frames are ready yet";
  }
}

function scheduleAutoPreview({ delay = 450, reason = "settings changed" } = {}) {
  if (!state.video || state.exportInFlight) return;
  if (!isEngineAvailable()) {
    updateEngineUi();
    return;
  }
  if (state.previewInFlight && state.engine !== "hypir") {
    cancelActivePreviewRequest();
  }
  const selectedLabel = engineLabels[state.engine] || state.engine;
  const effectiveDelay = state.engine === "hypir" ? delay : Math.max(delay, 650);
  state.previewDirty = true;
  window.clearTimeout(state.previewTimer);
  setActivity(`${selectedLabel} preview queued: ${reason}`, 0.12);
  state.previewTimer = window.setTimeout(() => {
    state.previewTimer = null;
    runAutoPreview();
  }, effectiveDelay);
}

async function runAutoPreview({ manual = false } = {}) {
  if (!state.video) return;
  if (state.previewInFlight) {
    if (state.engine === "hypir" && !manual) return;
    cancelActivePreviewRequest();
  }
  if (!isEngineAvailable()) {
    updateEngineUi();
    return;
  }
  state.previewDirty = false;
  state.previewInFlight = true;
  setExportEnabled(false);
  const version = ++state.previewVersion;
  const selectedLabel = engineLabels[state.engine] || state.engine;
  setActivity(state.engine === "hypir" ? "Enhancing preview" : `Running ${selectedLabel} preview`, 0.2);
  let timedOut = false;
  let timeoutId = null;
  try {
    const controller = new AbortController();
    state.previewController = controller;
    const timeoutMs = previewTimeoutMsByEngine[state.engine] || 2 * 60 * 1000;
    timeoutId = window.setTimeout(() => {
      timedOut = true;
      controller.abort();
      fetch("/api/preview/cancel", { method: "POST" }).catch(() => {});
    }, timeoutMs);
    const body = {
      videoId: state.video.id,
      seconds: Number(els.timeline.value),
      ...collectSettings(),
    };
    const preview = await api("/api/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (version !== state.previewVersion || state.previewDirty) return;
    const passText = preview.passes === 2 ? " | 2 passes" : "";
    const modeText = String(preview.result.previewMode || "").endsWith("_safe") ? " | safe preview" : "";
    const clipText = preview.result.clipSeconds ? ` | ${preview.result.clipSeconds}s clip` : "";
    showFramePair({
      originalUrl: `${preview.originalUrl}?t=${Date.now()}`,
      enhancedUrl: `${preview.enhancedUrl}?t=${Date.now()}`,
      originalInfo: formatTime(preview.seconds),
      enhancedInfo: `${preview.result.width}x${preview.result.height}${modeText}${clipText} | seed ${preview.result.seed}${passText}`,
    });
    rememberEngineMetric(state.engine, {
      averageFrameSeconds: preview.result.averageFrameSeconds,
      resolution: preview.result.metricResolution || `${preview.result.width}x${preview.result.height}`,
      source: "preview",
    });
    setActivity("Preview ready", 1);
    await refreshStatus();
  } catch (error) {
    if (timedOut && version === state.previewVersion) {
      setActivity("Preview timed out and was cancelled", 0);
    } else if (error.name !== "AbortError" && !String(error.message || "").toLowerCase().includes("cancelled")) {
      setActivity(`Preview failed: ${error.message}`, 0);
    }
  } finally {
    if (timeoutId !== null) {
      window.clearTimeout(timeoutId);
    }
    if (version === state.previewVersion) {
      state.previewInFlight = false;
      state.previewController = null;
      setExportEnabled(true);
      if (state.previewDirty && !state.exportInFlight) {
        scheduleAutoPreview({ delay: 50, reason: "latest change" });
      }
    }
  }
}

async function startExport() {
  if (!state.video) return;
  window.clearTimeout(state.previewTimer);
  if (state.previewInFlight) {
    cancelActivePreviewRequest();
  }
  clearFramePlayback();
  stopReadyPlayback();
  state.readyFrameEvents = [];
  state.readyFrameSeqs = new Set();
  state.exportInFlight = true;
  state.partialExportInFlight = false;
  state.lastExportFramesDone = 0;
  state.lastExportFramesTotal = 0;
  state.previewDirty = false;
  setExportEnabled(false);
  updateReadyPlaybackButton();
  saveLocalState();
  els.downloadLink.hidden = true;
  els.partialDownloadLink.hidden = true;
  els.partialDownloadLink.removeAttribute("href");
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
    updatePartialExportButton();
    saveLocalState();
    pollJob(job.id);
  } catch (error) {
    setActivity(`Export failed: ${error.message}`, 0);
    state.exportInFlight = false;
    setExportEnabled(true);
  }
}

async function startAdapterTraining() {
  if (!state.video || state.adapterInFlight) return;
  window.clearTimeout(state.previewTimer);
  state.previewDirty = false;
  state.adapterInFlight = true;
  setExportEnabled(false);
  saveLocalState();
  els.frameProgressText.textContent = "Adapter: preparing";
  els.etaText.textContent = "ETA: long running";
  setActivity("Starting film adapter training", 0.02);
  try {
    const job = await api("/api/adapters/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        videoId: state.video.id,
        prompt: els.prompt.value || "film-specific restoration, natural detail, consistent texture",
        quality: els.adapterQuality.value,
      }),
    });
    state.currentAdapterJobId = job.id;
    saveLocalState();
    pollAdapterJob(job.id);
  } catch (error) {
    setActivity(`Film adapter failed to start: ${error.message}`, 0);
    state.adapterInFlight = false;
    state.currentAdapterJobId = null;
    saveLocalState();
    setExportEnabled(true);
  }
}

async function cancelAdapterTraining() {
  if (!state.currentAdapterJobId) return;
  els.cancelAdapterButton.disabled = true;
  setActivity("Stopping film adapter training", null);
  try {
    await api(`/api/jobs/${state.currentAdapterJobId}/cancel`, { method: "POST" });
  } catch (error) {
    setActivity(`Stop failed: ${error.message}`, null);
    els.cancelAdapterButton.disabled = false;
  }
}

async function deleteSelectedAdapter() {
  const selected = selectedAdapterRecord();
  const rootAdapter = selectedRootAdapterRecord();
  if (!selected || !rootAdapter || !canDeleteSelectedAdapter()) return;

  const rootId = rootAdapter.parentAdapterId || rootAdapter.id.split("@step-")[0];
  const confirmed = window.confirm(
    `Delete film adapter "${rootAdapter.name || rootId}" and all of its checkpoints, patches, sampled frames, and logs? This cannot be undone.`
  );
  if (!confirmed) return;

  els.deleteAdapterButton.disabled = true;
  setActivity("Deleting film adapter", null);
  try {
    const result = await api(`/api/adapters/${encodeURIComponent(selected.id)}`, { method: "DELETE" });
    await loadAdapters("base");
    saveLocalState();
    setActivity(
      `Deleted adapter: ${result.filesDeleted} files and ${result.directoriesDeleted} folders (${formatBytes(result.bytesFreed)} freed).`,
      null
    );
    if (state.video) {
      scheduleAutoPreview({ delay: 300, reason: "film adapter deleted" });
    }
    await refreshStatus();
  } catch (error) {
    setActivity(`Delete adapter failed: ${error.message}`, null);
  } finally {
    setExportEnabled(true);
  }
}

async function deleteAllAdapters() {
  const adapterCount = rootAdapterRecords().length;
  if (!canDeleteAllAdapters()) return;
  const confirmed = window.confirm(
    `Delete all ${adapterCount} film adapter${adapterCount === 1 ? "" : "s"} and every checkpoint, patch, sampled frame, and log under work/adapters? Base HYPIR will stay. This cannot be undone.`
  );
  if (!confirmed) return;

  els.deleteAllAdaptersButton.disabled = true;
  els.deleteAdapterButton.disabled = true;
  setActivity("Deleting all film adapters", null);
  try {
    const result = await api("/api/adapters", { method: "DELETE" });
    await loadAdapters("base");
    saveLocalState();
    setActivity(
      `Deleted ${result.adaptersDeleted} adapters: ${result.filesDeleted} files and ${result.directoriesDeleted} folders (${formatBytes(result.bytesFreed)} freed).`,
      null
    );
    if (state.video) {
      scheduleAutoPreview({ delay: 300, reason: "film adapters deleted" });
    }
    await refreshStatus();
  } catch (error) {
    setActivity(`Delete all adapters failed: ${error.message}`, null);
  } finally {
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

async function savePartialExport() {
  if (!state.currentJobId || state.partialExportInFlight) return;
  state.partialExportInFlight = true;
  updatePartialExportButton();
  setActivity("Saving partial video", null);
  try {
    const result = await api(`/api/jobs/${state.currentJobId}/partial-export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        crf: Number(els.crf.value),
        encoder: els.encoder.value,
      }),
    });
    const totalText = result.framesTotal ? ` / ${result.framesTotal}` : "";
    els.partialDownloadLink.href = result.outputUrl;
    els.partialDownloadLink.textContent = `Download partial (${result.framesDone}${totalText} frames, ${formatTime(result.durationSeconds)})`;
    els.partialDownloadLink.hidden = false;
    setActivity(`Partial saved: ${result.framesDone}${totalText} frames with ${result.encoder}`, null);
  } catch (error) {
    setActivity(`Partial save failed: ${error.message}`, null);
  } finally {
    state.partialExportInFlight = false;
    setExportEnabled(!state.exportInFlight && !state.previewInFlight && !state.adapterInFlight);
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
  if (state.exportInFlight || state.previewInFlight || state.partialExportInFlight) return;
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
  if (Number.isFinite(frame.seconds)) {
    els.timeline.value = String(frame.seconds);
    els.currentTime.textContent = formatTime(frame.seconds);
  }
  showFramePair({
    originalUrl,
    enhancedUrl,
    originalInfo: `frame ${frame.frameIndex} | ${formatTime(frame.seconds)}`,
    enhancedInfo: frame.cached
      ? `frame ${frame.frameIndex} / ${frame.framesTotal} | cached`
      : `frame ${frame.frameIndex} / ${frame.framesTotal} | generated`,
  });
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
      rememberReadyFrames(result.frames);
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
  if (Number.isFinite(job.currentFrameSeconds)) {
    els.timeline.value = String(job.currentFrameSeconds);
    els.currentTime.textContent = formatTime(job.currentFrameSeconds);
  }
  showFramePair({
    originalUrl,
    enhancedUrl,
    originalInfo: job.currentFrameIndex ? `frame ${job.currentFrameIndex}` : "source frame",
    enhancedInfo: job.framesTotal ? `frame ${job.currentFrameIndex} / ${job.framesTotal}` : "enhanced frame",
  });
}

function updateJobUi(job) {
  setActivity(job.message, job.progress);
  if (job.averageFrameSeconds) {
    rememberEngineMetric(job.engine || state.engine, {
      averageFrameSeconds: job.averageFrameSeconds,
      resolution: job.metricResolution,
      source: job.metricSource || "export",
    });
  }
  state.lastExportFramesDone = Number(job.partialFramesReady ?? job.framesDone) || 0;
  state.lastExportFramesTotal = Number(job.framesTotal) || 0;
  updatePartialExportButton();
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

function updateAdapterJobUi(job) {
  setActivity(job.message, job.progress);
  if (job.framesTotal) {
    els.frameProgressText.textContent = `Adapter: ${job.framesDone} training patches`;
  } else {
    els.frameProgressText.textContent = "Adapter: preparing";
  }
  els.etaText.textContent = "ETA: long running";
}

function pollAdapterJob(jobId) {
  if (state.adapterTimer) clearInterval(state.adapterTimer);
  state.adapterTimer = setInterval(async () => {
    try {
      const job = await api(`/api/jobs/${jobId}`);
      state.adapterPollFailures = 0;
      updateAdapterJobUi(job);
      if (job.status === "done") {
        clearInterval(state.adapterTimer);
        state.adapterTimer = null;
        state.adapterInFlight = false;
        state.currentAdapterJobId = null;
        if (job.adapterId) {
          await loadAdapters(job.adapterId);
          saveLocalState();
        }
        els.frameProgressText.textContent = job.adapterName
          ? `Adapter: ${job.adapterName}`
          : "Adapter: ready";
        els.etaText.textContent = "ETA: --";
        setExportEnabled(true);
        await refreshStatus();
      }
      if (job.status === "error" || job.status === "cancelled") {
        clearInterval(state.adapterTimer);
        state.adapterTimer = null;
        setActivity(job.status === "cancelled" ? job.message : `Film adapter failed: ${job.error || job.message}`, job.progress);
        state.adapterInFlight = false;
        state.currentAdapterJobId = null;
        saveLocalState();
        setExportEnabled(true);
      }
    } catch (error) {
      if (error.status === 404) {
        clearInterval(state.adapterTimer);
        state.adapterTimer = null;
        setActivity("Film adapter job is no longer available", 0);
        state.adapterInFlight = false;
        state.currentAdapterJobId = null;
        saveLocalState();
        setExportEnabled(true);
        return;
      }
      state.adapterPollFailures += 1;
      setActivity(`Film adapter polling interrupted (${state.adapterPollFailures}). Retrying: ${error.message}`);
    }
  }, 1200);
}

function pollJob(jobId) {
  if (state.jobTimer) clearInterval(state.jobTimer);
  state.jobTimer = setInterval(async () => {
    try {
      const job = await api(`/api/jobs/${jobId}`);
      state.jobPollFailures = 0;
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
      if (error.status === 404) {
        clearInterval(state.jobTimer);
        state.jobTimer = null;
        setActivity("Export job is no longer available", 0);
        state.exportInFlight = false;
        state.currentJobId = null;
        saveLocalState();
        setExportEnabled(true);
        return;
      }
      state.jobPollFailures += 1;
      setActivity(`Job polling interrupted (${state.jobPollFailures}). Retrying: ${error.message}`);
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
els.previewButton.addEventListener("click", () => runAutoPreview({ manual: true }));
els.playReadyButton.addEventListener("click", toggleReadyPlayback);
els.cancelExportButton.addEventListener("click", cancelExport);
els.partialExportButton.addEventListener("click", savePartialExport);
els.trainAdapterButton.addEventListener("click", startAdapterTraining);
els.cancelAdapterButton.addEventListener("click", cancelAdapterTraining);
els.deleteAdapterButton.addEventListener("click", deleteSelectedAdapter);
els.deleteAllAdaptersButton.addEventListener("click", deleteAllAdapters);
els.openOutputFolderButton.addEventListener("click", openOutputFolder);
els.clearGeneratedButton.addEventListener("click", clearGeneratedFiles);
els.refreshStatusButton.addEventListener("click", refreshStatus);
els.engineCards.forEach((card) => {
  card.addEventListener("click", () => {
    const engine = card.dataset.engine;
    if (!engine || card.classList.contains("disabled")) return;
    setEngine(engine);
  });
});
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
els.adapterSelect.addEventListener("change", () => {
  updateDeleteAdapterButton();
  saveLocalState();
  scheduleAutoPreview({ delay: 300, reason: "film adapter changed" });
});
els.secondPass.addEventListener("change", () => {
  saveLocalState();
  scheduleAutoPreview({ delay: 300, reason: "second pass changed" });
});
els.adapterQuality.addEventListener("change", saveLocalState);
els.seed.addEventListener("input", () => {
  saveLocalState();
  scheduleAutoPreview();
});
els.prompt.addEventListener("input", () => {
  saveLocalState();
  scheduleAutoPreview();
});
[
  els.seedvr2BatchSize,
  els.seedvr2TemporalOverlap,
  els.seedvr2ChunkSize,
  els.seedvr2ColorCorrection,
  els.seedvr2PreviewSize,
  els.flashvsrVariant,
  els.flashvsrSparseRatio,
  els.flashvsrLocalRange,
  els.flashvsrPreviewCap,
].forEach((element) => {
  element.addEventListener("change", () => {
    saveLocalState();
    scheduleAutoPreview();
  });
});
els.encoder.addEventListener("change", saveLocalState);
els.crf.addEventListener("input", () => {
  els.crfValue.textContent = els.crf.value;
  saveLocalState();
});

els.comparisonBox.addEventListener("pointerdown", beginComparisonDrag);
window.addEventListener("pointermove", moveComparisonDrag);
window.addEventListener("pointerup", endComparisonDrag);
els.comparisonHandle.addEventListener("keydown", (event) => {
  if (!els.comparisonBox.classList.contains("comparison-active")) return;
  if (event.key === "ArrowLeft") {
    event.preventDefault();
    setComparisonPosition(comparisonPosition() - 3);
  }
  if (event.key === "ArrowRight") {
    event.preventDefault();
    setComparisonPosition(comparisonPosition() + 3);
  }
});

state.engineMetrics = readEngineMetrics();
updateScaleMode();
setupTooltips();
resetGeneratedUi();
refreshStatus();
restoreSession();
