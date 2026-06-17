# CleanVideo

Local video restoration UI for CUDA-first enhancement, video-native super-resolution, live preview, H.264 export, and HYPIR film-adapter experiments.

![CleanVideo live export preview](docs/assets/cleanvideo-live-export.png)

## Current feature set

- Five selectable engines in the UI: `FlashVSR`, `SeedVR2`, `DOVE`, `SUPIR`, and `HYPIR`.
- `FlashVSR` is the default engine and runs final exports through the continuous official v1.1 Tiny/Tiny Long streaming path.
- `SeedVR2` provides video-native restoration through the local SeedVR2 CLI with batch, overlap, chunk, and color-correction controls.
- `DOVE` is wired as a video-native one-step VSR CLI path through the local `external/DOVE` checkout and `.venv-dove`.
- `SUPIR` is wired as an SDXL per-frame folder inference path through the local `external/SUPIR` checkout and `.venv-supir`.
- `HYPIR` provides per-frame SD2 restoration, frame caching, partial export, export-time temporal stabilization, and local LoRA film adapters.
- Upload, seek, auto-preview, draggable before/after comparison, export progress, ETA, frame playback, cancellation, output-folder opening, and destructive `Clean All` work cleanup.
- Status-aware engine cards show readiness and per-video speed metrics saved in browser `localStorage`.
- HYPIR and partial exports use `h264_nvenc` when available and fall back to `libx264`. Video-native engines write their model output first, then remux it with the source audio.

## Quick start

On Windows, double-click:

```text
Start-CleanVideo.cmd
```

The launcher runs:

1. `scripts/start.ps1 -RestartWhenIdle`
2. `scripts/setup.ps1 -IfNeeded`
3. `scripts/launch.ps1`

It starts the local server at <http://127.0.0.1:8765> and opens the browser. If an idle CleanVideo server is already running on the port, the launcher restarts it so the latest local code is used. If a job is active, `-RestartWhenIdle` reuses the running server.

From Codex, use the Run menu actions:

- `Launch CleanVideo`: setup-if-needed, restart only when idle, open browser.
- `Force Restart CleanVideo`: setup-if-needed, force restart the local server.
- `Run Server Only`: setup-if-needed, run without opening the browser.
- `Setup / Reinstall Dependencies`: run the base setup script directly.

Server-only mode:

```powershell
.\run.ps1
```

Reinstall or repair base dependencies:

```powershell
.\scripts\setup.ps1
```

Only install missing base items:

```powershell
.\scripts\setup.ps1 -IfNeeded
```

## Setup scope

`scripts/setup.ps1` bootstraps the base HYPIR application path:

- `uv`
- Python 3.10 virtual environment at `.venv`
- `requirements-inference.txt`, including PyTorch CUDA wheels
- FFmpeg and ffprobe
- `external/HYPIR`
- `external/SUPIR`
- `models/hypir/HYPIR_sd2.pth`
- `models/stable-diffusion-2-1-base`

SeedVR2, FlashVSR, DOVE, and SUPIR are active engine integrations in the code, but their large optional runtimes are checked by `/api/status` rather than recreated by the base setup script. If their paths, models, or Python environments are missing, the UI marks the engine blocked and the API returns a specific readiness error. This checkout has DOVE and SUPIR configured locally; a fresh machine still needs their optional runtimes and weights installed before those cards turn ready.

Check runtime status:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/api/status | ConvertTo-Json -Depth 6
```

## Engine matrix

| Engine | State | Best use | Preview | Export | Adapters | Main limits |
| --- | --- | --- | --- | --- | --- | --- |
| `FlashVSR` | Active when WSL/Block-Sparse-Attention/models are ready | Fast video-native super-resolution | 21-frame safe clip around the playhead, with timeout and safe-resolution retry | Continuous streaming Tiny Long/Tiny, source audio remuxed | Not available | No final chunked render, no native downscale, 512px minimum longest side, 4x native cap, GPU/WSL pixel-budget guard |
| `SeedVR2` | Active when `.venv-seedvr2`, CLI, DiT, and VAE are ready | Video-native restoration with temporal context | Single extracted frame through SeedVR2 safe preview mode | Whole-video CLI export, source audio remuxed | Not available | Heavy CUDA stack, no partial export, no supported local video-to-adapter workflow |
| `DOVE` | Active when `.venv-dove`, `external/DOVE/inference_script.py`, and the DOVE/CogVideoX component folders including `vae` are ready | One-step diffusion VSR through CogVideoX-based inference | Short safe clip around the playhead | Whole-video CLI export, source audio remuxed | Not available | Optional runtime only, integer upscale path, no partial export |
| `SUPIR` | Active when `.venv-supir`, SUPIR config, SDXL/SUPIR checkpoints, and local CLIP paths are ready | SDXL per-frame restoration comparison path | Single-frame SUPIR folder inference | PNG frame cache, then H.264 encode | Not available | Optional runtime only, no temporal stabilization, no partial export |
| `HYPIR` | Active when CUDA, HYPIR weights, and SD2 base model are ready | High-quality per-frame restoration and film-specific experiments | Single-frame HYPIR pass | PNG frame cache, optional temporal stabilization, H.264 encode | Local LoRA film adapters | Slower on long videos, temporal consistency is export-centric |

## FlashVSR

CleanVideo integrates the official FlashVSR v1.1 WanVSR path through `scripts/flashvsr_cli.py` and `app/flashvsr_engine.py`.

Required layout:

- `external/FlashVSR`
- `external/Block-Sparse-Attention`
- `external/FlashVSR/examples/WanVSR/FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors`
- `external/FlashVSR/examples/WanVSR/FlashVSR-v1.1/LQ_proj_in.ckpt`
- `external/FlashVSR/examples/WanVSR/FlashVSR-v1.1/TCDecoder.ckpt`
- `external/FlashVSR/examples/WanVSR/FlashVSR-v1.1/Wan2.1_VAE.pth`
- Optional Windows env: `.venv-flashvsr`
- Active WSL bridge: `scripts/flashvsr_wsl_python.sh`
- WSL Python: `/home/user/.cleanvideo/flashvsr-wsl/.venv/bin/python`
- WSL CUDA prefix: `/home/user/.cleanvideo/cuda-12.4`

The Windows-native Block-Sparse-Attention build is currently blocked on this machine by a CUDA/Visual Studio compiler access violation. The active backend is WSL2 Ubuntu with the LCSA/block-sparse extension built on Linux ext4. Dense/community fallback is intentionally not enabled because it can reduce official FlashVSR quality.

Runtime behavior:

- UI variants: `Tiny Long` and `Tiny`.
- The API type still knows `full`, but final render/export blocks it because the Full path would require chunked render. Chunked FlashVSR is allowed only for preview/probing, not final render/export.
- Final export passes `--streaming`, `--total_frames`, `--fps`, `--live_original`, and `--live_enhanced` into the CLI.
- Live export preview overwrites a single latest source JPEG and a single latest enhanced JPEG instead of saving every FlashVSR frame as a preview image.
- FlashVSR output is remuxed with the source audio after the model pass.
- Transient Python import or torch-load corruption is retried.
- WSL mount warnings are ignored when they are unrelated noise.
- CUDA OOM, exit code 9, and exit code 11 are surfaced as user-facing resolution/VRAM guidance.

Native export guards:

- Variant must be `tiny_long` or `tiny`.
- Requested output longest side must be at least `512`.
- Requested output cannot be smaller than the source longest side.
- Requested output cannot exceed the local 4x FlashVSR scale cap.
- Estimated output pixels per frame must fit the safe native budget:
  - unknown or under 24 GB VRAM: about 2.1 MP/frame
  - 24 GB VRAM: about 3.0 MP/frame
  - 32 GB VRAM: about 8.5 MP/frame
  - 48 GB+ VRAM: about 12.5 MP/frame

If a request exceeds those limits, the export is blocked before the model runs and the UI tells the user to lower Resolution/Scale Mode or choose another engine.

## SeedVR2

CleanVideo integrates SeedVR2 through `app/seedvr2_engine.py` and the local `external/SeedVR2_VideoUpscaler/inference_cli.py`.

Required layout:

- `.venv-seedvr2/Scripts/python.exe`
- `external/SeedVR2_VideoUpscaler/inference_cli.py`
- `models/seedvr2/seedvr2_ema_3b_fp8_e4m3fn.safetensors`
- `models/seedvr2/ema_vae_fp16.safetensors`

Runtime behavior:

- Preview mode extracts one safe frame and calls SeedVR2 with `batch_size=1`, `chunk_size=0`, and `load_cap=1`.
- Export mode passes the original video to the SeedVR2 CLI, then remuxes the result with the source audio.
- The target output can come from a scale factor, a fixed preset, or custom longest side.
- SeedVR2 is video-native, so it does not expose HYPIR-style cached PNG partial export.

UI controls:

- `Batch Size`: 1, 5, 9, 13, 17, 21, 25. Larger batches can improve consistency and speed but use more VRAM.
- `Temporal Overlap`: 0-5 blended frames between batches/chunks.
- `Chunk Size`: 0, 85, 170, 340. `170` is the default bounded-memory long-video setting; `0` processes the whole video at once.
- `Color Correction`: `Lab`, `Wavelet`, `Wavelet Adaptive`, `HSV`, `AdaIN`, or `None`.

SeedVR2 public code has training configs, but this app does not currently have a supported local video-to-adapter workflow comparable to HYPIR LoRA adapters.

## DOVE

CleanVideo integrates DOVE through `app/dove_engine.py` and the local `external/DOVE/inference_script.py`.

Required layout:

- `.venv-dove/Scripts/python.exe`
- `external/DOVE/inference_script.py`
- `external/DOVE/pretrained_models/DOVE/model_index.json`
- DOVE component folders: `scheduler`, `text_encoder`, `tokenizer`, `transformer`
- CogVideoX VAE folder under `external/DOVE/pretrained_models/DOVE/vae`
- Optional empty-prompt embedding under `external/DOVE/pretrained_models/prompt_embeddings`

Runtime behavior:

- Preview mode extracts a short safe clip around the playhead and renders a representative enhanced frame.
- Export mode calls the DOVE CLI on the source video, then remuxes source audio.
- DOVE uses integer upscale factors, so custom longest-side targets are rounded up to a 1-4x factor.
- DOVE does not use HYPIR film adapters and does not support partial export.

UI controls:

- `Chunk Length`: temporal chunk size for low-memory export; `0` processes the clip as one chunk.
- `Overlap`: overlapping frames between chunks; the backend clamps it below `Chunk Length`.
- `Offload`: enables DOVE sequential CPU offload for lower VRAM use at lower speed.

## SUPIR

CleanVideo integrates SUPIR through `app/supir_engine.py` and the local `external/SUPIR/test.py`.

Required layout:

- `.venv-supir/Scripts/python.exe`
- `external/SUPIR/test.py`
- `external/SUPIR/options/SUPIR_v0.yaml`
- `external/SUPIR/CKPT_PTH.py`
- `models/supir/sd_xl_base_1.0_0.9vae.safetensors`
- `models/supir/SUPIR-v0Q.ckpt`
- `models/supir/SUPIR-v0F.ckpt`
- local CLIP paths configured in `CKPT_PTH.py`

Runtime behavior:

- Preview mode enhances a single extracted frame through SUPIR folder inference.
- Export mode extracts all frames, enhances missing PNG frames through SUPIR, then encodes H.264.
- The backend runs SUPIR with `--no_llava` by default; LLaVA weights can be present locally but are not required for the default app path.
- On Windows, the local SUPIR config uses native PyTorch attention instead of `xformers/triton`.
- SUPIR is slow to cold-start because each subprocess loads SDXL/SUPIR weights; first-frame smoke tests can still take many minutes.

UI controls:

- `Sign`: `Q` quality checkpoint or `F` fidelity checkpoint.
- `Color Fix`: `Wavelet`, `AdaIn`, or `None`.
- `Steps`: diffusion steps; `50` is the quality default and `1-4` is useful for setup validation.
- `Min Size`: short-side minimum before SUPIR inference; `1024` is the quality default and lower values reduce VRAM/time for smoke tests.

## HYPIR

HYPIR is the base per-frame restoration path in `app/hypir_engine.py`.

Required layout:

- `.venv/Scripts/python.exe`
- `external/HYPIR/HYPIR/enhancer/sd2.py`
- `models/hypir/HYPIR_sd2.pth`
- `models/stable-diffusion-2-1-base`

Runtime behavior:

- HYPIR loads SD2 through `SD2Enhancer` and keeps the active weight in memory until a different base/adapter weight is selected or an adapter delete unloads it.
- Preview runs one frame at the selected timeline position.
- Export extracts source frames, enhances missing PNG frames into `work/cache/<cache-key>/enhanced`, encodes H.264, and schedules intermediate cleanup.
- Restarting the same HYPIR export reuses valid cached enhanced frames.
- `Save Partial` builds an MP4 from the contiguous ready enhanced frame prefix.
- `Play Ready` plays generated PNG frame events at the source FPS in the preview pane.

HYPIR controls:

- `Prompt`: passed to HYPIR for the restoration look.
- `Seed`: fixed seed for repeatable preview/export, or `-1` for random.
- `Patch Size`: 512, 768, 1024.
- `Stride`: 256, 384, 512, 768. If stride exceeds patch size, the UI clamps it to patch size.
- `Temporal`: `Off`, `Light`, `Medium`, `Strong`, `Extra Strong`.

The `0.5x` scale factor is a pre-HYPIR Lanczos downscale. The image is resized to half width/height first, then HYPIR runs at `1x` so the shrink happens only once.

## Temporal consistency

Temporal consistency is implemented in `app/temporal_stabilizer.py` and is currently HYPIR export-only.

After each HYPIR frame is enhanced, CleanVideo:

1. Computes optical flow between the previous source frame and the current source frame.
2. Warps the previous enhanced output toward the current frame.
3. Blends stable regions into the current enhanced frame.
4. Skips blending across detected scene cuts.

Profiles:

| Mode | Blend | Notes |
| --- | ---: | --- |
| `Off` | 0.00 | Original independent-frame behavior |
| `Light` | 0.16 | Mild flicker reduction |
| `Medium` | 0.26 | Default balance |
| `Strong` | 0.38 | More stable, more risk of motion softness |
| `Extra Strong` | 0.54 | Steadiest, highest risk of ghosting/softening |

Preview only processes one frame, so it cannot show the real temporal benefit. Verify temporal changes on exported frames.

## Film adapters

Film adapters are HYPIR-only local LoRA fine-tunes managed by `app/film_adapter.py` and `app/main.py`.

Workflow:

1. Upload/select a source video.
2. Choose `HYPIR`.
3. Select `Adapter Quality`.
4. Click `Create Film Adapter`.
5. The app samples frames, selects usable/non-duplicate frames, crops 512x512 training patches, writes a HYPIR training YAML, and launches `accelerate`.
6. The resulting adapter is saved under `work/adapters/<job-id>`.

Quality presets:

| Preset | Selected frames | Patches/frame | Train steps |
| --- | ---: | ---: | ---: |
| `Fast` | 64-160 | 1 | 240-600 |
| `High` | 192-768 | 2 | 900-3000 |
| `Extra` | 480-1600 | 2 | 1500-6000 |

Longer videos automatically select more frames up to the preset cap. The sampler filters very dark, low-detail, or near-duplicate frames before writing patches.

Recovery and cleanup:

- Training checkpoints are written under the adapter root.
- If the trainer exits before the target step but usable checkpoints exist, CleanVideo rewrites `resume_from_checkpoint` to the latest `checkpoint-N` directory and retries up to 3 times.
- If a final checkpoint already reached the target step, the adapter is treated as complete even if the trainer exited non-zero afterward.
- After success, temporary source samples, patches, dataset files, logs, old checkpoints, optimizer files, and EMA files are removed.
- The kept artifact is the final `state_dict.pth` plus `adapter.json`.

Adapter controls:

- `Adapter`: choose `Base HYPIR` or a trained adapter.
- Trash button: delete the selected adapter root.
- `Delete All Adapters`: delete every film adapter under `work/adapters`; `Base HYPIR` is protected.
- `Second Pass: Base after adapter`: first run the selected adapter, then run Base HYPIR at 1x as a refinement pass. This is slower and experimental.

## Resolution and export settings

Resolution modes:

- `Scale factor`: 0.5x, 1x, 2x, 3x, 4x.
- `720p`: longest side 1280.
- `Full HD`: longest side 1920.
- `4K`: longest side 3840.
- `8K`: longest side 7680.
- `Target longest side`: custom 256-8192.

Export settings:

- `Resource Mode: Responsive`: lowers worker/ffmpeg process priority so foreground apps stay usable.
- `Resource Mode: Maximum`: normal worker priority.
- `Encoder: Auto`: prefer NVIDIA NVENC when available.
- `Encoder: NVIDIA NVENC`: use `h264_nvenc`.
- `Encoder: CPU libx264`: use CPU encode.
- `Quality CRF/CQ`: 12-32. Lower means higher quality and larger files. NVENC treats it as CQ; libx264 treats it as CRF.

The encoder and CRF/CQ controls apply to HYPIR full exports and HYPIR partial exports. FlashVSR and SeedVR2 exports are model-produced MP4 files that CleanVideo remuxes with source audio.

## Work directories

All generated working files live under `work`:

- `work/uploads`: uploaded source videos and metadata.
- `work/previews`: preview frame pairs and temporary preview clips.
- `work/exports`: final and partial MP4 outputs.
- `work/jobs`: persisted job records.
- `work/cache`: HYPIR frame caches and native-engine intermediate outputs.
- `work/partials`: temporary partial-export staging.
- `work/adapters`: HYPIR film adapters.

`Clean All` deletes everything under `work`, including uploads, exports, previews, caches, partial videos, job records, and film adapters. It recreates the empty working folder layout afterward. Base models and source checkouts outside `work` are not deleted.

Cleanup is blocked while an export, adapter job, partial export, or active preview is still running. If previews are still finishing, the server cancels preview workers and asks the user to retry cleanup after they stop.

## API surface

Health and status:

- `GET /api/health`
- `GET /api/status`

Videos:

- `GET /api/videos`
- `POST /api/videos`

Preview:

- `POST /api/preview`
- `POST /api/preview/cancel`

Exports and jobs:

- `POST /api/export`
- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/frames`
- `POST /api/jobs/{job_id}/partial-export`
- `POST /api/jobs/{job_id}/cancel`

Adapters:

- `GET /api/adapters`
- `POST /api/adapters/train`
- `DELETE /api/adapters/{adapter_id}`
- `DELETE /api/adapters`

Maintenance:

- `POST /api/open-output-folder`
- `POST /api/cleanup-generated`

## Installed and optional source layout

Core:

- `app/main.py`: FastAPI app, job state, engine routing, export orchestration, cleanup.
- `app/hypir_engine.py`: HYPIR SD2 wrapper.
- `app/seedvr2_engine.py`: SeedVR2 CLI wrapper.
- `app/flashvsr_engine.py`: FlashVSR Windows/WSL wrapper and status probes.
- `app/dove_engine.py`: DOVE video-native CLI wrapper.
- `app/supir_engine.py`: SUPIR folder-image CLI wrapper.
- `app/film_adapter.py`: HYPIR adapter dataset/training helpers.
- `app/temporal_stabilizer.py`: optical-flow export stabilization.
- `app/video_ops.py`: FFmpeg probing, frame extraction, encode, mux.
- `static/index.html`, `static/app.js`, `static/styles.css`: local UI.

External runtimes used by this checkout:

- `external/HYPIR`: upstream HYPIR source.
- `external/SUPIR`: upstream SUPIR source used when `.venv-supir` and checkpoints are configured.
- `external/DOVE`: optional upstream DOVE source expected by the DOVE engine.
- `external/SeedVR`: upstream SeedVR source/reference checkout.
- `external/SeedVR2_VideoUpscaler`: local SeedVR2 CLI used by the app.
- `external/FlashVSR`: official FlashVSR source.
- `external/Block-Sparse-Attention`: FlashVSR LCSA/block-sparse dependency.
- `.venv`: main CleanVideo/HYPIR environment.
- `.venv-seedvr2`: isolated SeedVR2 environment.
- `.venv-flashvsr`: isolated Windows FlashVSR environment used for status/probing when applicable.
- `.venv-dove`: optional DOVE Python 3.11 environment.
- `.venv-supir`: optional SUPIR Python environment.

Models:

- `models/hypir/HYPIR_sd2.pth`
- `models/stable-diffusion-2-1-base`
- `models/seedvr2/seedvr2_ema_3b_fp8_e4m3fn.safetensors`
- `models/seedvr2/ema_vae_fp16.safetensors`
- FlashVSR weights under `external/FlashVSR/examples/WanVSR/FlashVSR-v1.1`
- DOVE weights under `external/DOVE/pretrained_models/DOVE`, including CogVideoX `vae`
- SUPIR checkpoints and extracted CLIP/LLaVA assets under `models/supir`
- SUPIR checkpoint paths configured in `external/SUPIR/options/SUPIR_v0.yaml`
- SUPIR CLIP/LLaVA paths configured in `external/SUPIR/CKPT_PTH.py`

## Roadmap and evaluated engines

Active now:

- `FlashVSR`: default video-native streaming SR path.
- `SeedVR2`: video-native restoration path.
- `DOVE`: optional video-native one-step VSR CLI path.
- `SUPIR`: optional per-frame SDXL restoration path.
- `HYPIR`: per-frame restoration and film-adapter path.

Reserved or evaluated:

- `DiffVSR`, `STAR`, `MGLD-VSR`, `Upscale-A-Video`: relevant diffusion VSR projects, but heavier/slower than the current interactive target.
- `VEnhancer`: strong for AI-generated video enhancement and space-time upsampling, but official single-GPU requirements are too high for this local workflow today.
- `VideoGigaGAN`: quality-relevant, but not currently practical until public code/weights are suitable.
- `Real-ESRGAN`, `StableSR`, `ResShift`, `SeeSR`, `DiffBIR`, `PASD`, `OSEDiff`: image/frame restoration paths that do not currently justify integration ahead of the existing HYPIR lane.
- `Z-Image Turbo`, FLUX-style editors, and other modern generators: possible future separate backends, not drop-in replacements for HYPIR's SD2 enhancer/trainer contract.

## Known limitations

- HYPIR upstream licensing is non-commercial; check upstream license terms before commercial use.
- FlashVSR final render/export must stay on the continuous streaming path. Chunked final render is disabled because it can damage final temporal quality.
- FlashVSR's official quality path requires Block-Sparse-Attention/LCSA. Missing LCSA is treated as blocked rather than silently falling back to a lower-quality path.
- FlashVSR native export fails early when the requested output size exceeds the model scale cap or local VRAM/WSL budget.
- SeedVR2 and FlashVSR do not support `Save Partial`.
- Film adapters are experimental and can overfit noisy or narrow source material.
- `Clean All` is intentionally destructive for everything under `work`.

## Development checks

Run the focused unit tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```

Useful status checks while developing:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/api/health
Invoke-RestMethod http://127.0.0.1:8765/api/status | ConvertTo-Json -Depth 6
```

If setup cannot download a model automatically, it prints the exact manual URL and destination path. The current SD2 base model mirror is `Manojb/stable-diffusion-2-1-base`.
