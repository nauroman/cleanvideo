# CleanVideo

Local video enhancement UI for HYPIR with CUDA preview and H.264 export.

![CleanVideo live export preview](docs/assets/cleanvideo-live-export.png)

## Run

On Windows, double-click:

```text
Start-CleanVideo.cmd
```

It starts the local server if needed and opens <http://127.0.0.1:8765>.
If an idle CleanVideo server is already running, the launcher restarts it so the latest local code is used.
On first launch, it also runs setup automatically if required. The setup checks/install/downloads:

- `uv` and a Python 3.10 virtual environment.
- Python inference dependencies, including PyTorch CUDA wheels.
- FFmpeg / ffprobe for frame extraction and H.264 export.
- HYPIR and SUPIR source trees.
- HYPIR weights and the Stable Diffusion 2.1 base model files.

If an automatic download fails, the console prints the exact manual download URL and the local folder/file where it must be placed.

From Codex, use the Run menu action:

```text
Launch CleanVideo
```

Use `Force Restart CleanVideo` only after stopping any export you want to keep running.

Server-only mode:

```powershell
.\run.ps1
```

## What is installed

- `external/HYPIR`: XPixelGroup HYPIR checkout, created by setup.
- `external/SUPIR`: SUPIR checkout reserved for a later engine switch, created by setup.
- `models/hypir/HYPIR_sd2.pth`: HYPIR-SD2 LoRA weights, created by setup.
- `models/stable-diffusion-2-1-base`: Diffusers-format Stable Diffusion 2.1 base mirror, created by setup.
- `.venv`: Python 3.10 virtual environment with `torch 2.6.0+cu124`.

The app uses PyTorch CUDA for HYPIR inference and `h264_nvenc` for H.264 export when available. If NVENC fails, the backend falls back to `libx264`.

During export, enhanced frames are cached under `work/cache` using the selected video and HYPIR settings. If the app is stopped mid-export, starting the same export again reuses already enhanced frames instead of generating them again. The browser also stores the last selected server-side video id and UI settings in `localStorage`.

The `Temporal` export setting reduces frame-to-frame flicker by optical-flow warping the previous enhanced frame onto the current source frame and blending stable areas after each HYPIR pass. `Medium` is the default balance; `Strong` and `Extra Strong` can be steadier but may soften fast motion or add ghosting, and `Off` preserves the old independent-frame behavior.

The `0.5x` scale factor performs high-quality Lanczos downscaling before HYPIR enhancement, producing half-width and half-height output while giving the generator a smaller, cleaner input.

`Create Film Adapter` samples the selected video into 512x512 training patches, starts a local HYPIR LoRA fine-tune job, and saves the resulting adapter under `work/adapters`. The `Adapter Quality` preset controls sampling and training length: `Fast` uses 32 selected frames, 3 patches per frame, and 300 steps; `High` uses 80 frames, 5 patches, and 900 steps; `Extra` uses 128 frames, 7 patches, and 1200 steps. The sampler extracts extra candidate frames, filters very dark, low-detail, or near-duplicate frames, and keeps multiple checkpoints so the `Film Adapter` selector can preview earlier/later steps.

`Second Pass` can run `Base after adapter` for preview/export. This first enhances with the selected film adapter, then runs Base HYPIR at 1x as a refinement pass before temporal stabilization. It is experimental and much slower, but useful to compare when an adapter preserves film style well and Base HYPIR adds cleaner detail on top. Adapter training is experimental: it can take a long time, may overfit noisy footage, and `Clean All` does not delete completed adapters.

Use the trash button next to `Film Adapter` to delete a completed adapter, or `Delete All Adapters` to remove every film adapter at once. Deleting adapters removes checkpoints, patches, sampled frames, logs, and metadata under `work/adapters`; `Base HYPIR` cannot be deleted.

## Reinstall dependencies

```powershell
.\scripts\setup.ps1
```

To only install missing items:

```powershell
.\scripts\setup.ps1 -IfNeeded
```

HYPIR is licensed for non-commercial use by its upstream project. The official `stabilityai/stable-diffusion-2-1-base` Hugging Face repo was unavailable during setup, so the local base model was downloaded from a public diffusers-compatible mirror.
