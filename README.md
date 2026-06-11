# CleanVideo

Local video enhancement UI for HYPIR with CUDA preview and H.264 export.

## Run

```powershell
.\run.ps1
```

Open <http://127.0.0.1:8765>.

## What is installed

- `external/HYPIR`: XPixelGroup HYPIR checkout, created by setup.
- `external/SUPIR`: SUPIR checkout reserved for a later engine switch, created by setup.
- `models/hypir/HYPIR_sd2.pth`: HYPIR-SD2 LoRA weights, created by setup.
- `models/stable-diffusion-2-1-base`: Diffusers-format Stable Diffusion 2.1 base mirror, created by setup.
- `.venv`: Python 3.10 virtual environment with `torch 2.6.0+cu124`.

The app uses PyTorch CUDA for HYPIR inference and `h264_nvenc` for H.264 export when available. If NVENC fails, the backend falls back to `libx264`.

During export, enhanced frames are cached under `work/cache` using the selected video and HYPIR settings. If the app is stopped mid-export, starting the same export again reuses already enhanced frames instead of generating them again. The browser also stores the last selected server-side video id and UI settings in `localStorage`.

## Reinstall dependencies

```powershell
.\scripts\setup.ps1
```

HYPIR is licensed for non-commercial use by its upstream project. The official `stabilityai/stable-diffusion-2-1-base` Hugging Face repo was unavailable during setup, so the local base model was downloaded from a public diffusers-compatible mirror.
