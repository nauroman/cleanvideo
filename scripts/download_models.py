import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


ROOT = Path(__file__).resolve().parents[1]
HYPIR_DIR = ROOT / "models" / "hypir"
HYPIR_FILE = HYPIR_DIR / "HYPIR_sd2.pth"
BASE_MODEL_DIR = ROOT / "models" / "stable-diffusion-2-1-base"
BASE_MODEL_REQUIRED = [
    "scheduler",
    "tokenizer",
    "text_encoder",
    "unet",
    "vae",
    "model_index.json",
]
BASE_MODEL_PATTERNS = [
    "scheduler/*",
    "tokenizer/*",
    "text_encoder/*",
    "unet/*",
    "vae/*",
    "model_index.json",
]


def print_manual_hypir_help(exc: Exception) -> None:
    print("", file=sys.stderr)
    print("Could not download HYPIR weights automatically.", file=sys.stderr)
    print(f"Error: {exc}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Manual download:", file=sys.stderr)
    print("  URL: https://huggingface.co/lxq007/HYPIR/resolve/main/HYPIR_sd2.pth", file=sys.stderr)
    print(f"  Save as: {HYPIR_FILE}", file=sys.stderr)
    print("", file=sys.stderr)


def print_manual_base_model_help(exc: Exception) -> None:
    print("", file=sys.stderr)
    print("Could not download the Stable Diffusion 2.1 base model automatically.", file=sys.stderr)
    print(f"Error: {exc}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Manual download:", file=sys.stderr)
    print("  URL: https://huggingface.co/Manojb/stable-diffusion-2-1-base/tree/main", file=sys.stderr)
    print(f"  Destination folder: {BASE_MODEL_DIR}", file=sys.stderr)
    print("  Required items: scheduler, tokenizer, text_encoder, unet, vae, model_index.json", file=sys.stderr)
    print("", file=sys.stderr)


def download_hypir_weights() -> None:
    if HYPIR_FILE.exists():
        print(f"HYPIR weights already present: {HYPIR_FILE}")
        return

    HYPIR_DIR.mkdir(parents=True, exist_ok=True)
    try:
        hf_hub_download(
            repo_id="lxq007/HYPIR",
            filename="HYPIR_sd2.pth",
            local_dir=HYPIR_DIR,
        )
    except Exception as exc:
        print_manual_hypir_help(exc)
        raise RuntimeError("Automatic HYPIR weight download failed.") from exc


def download_base_model() -> None:
    if all((BASE_MODEL_DIR / item).exists() for item in BASE_MODEL_REQUIRED):
        print(f"Stable Diffusion base model already present: {BASE_MODEL_DIR}")
        return

    BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id="Manojb/stable-diffusion-2-1-base",
            local_dir=BASE_MODEL_DIR,
            allow_patterns=BASE_MODEL_PATTERNS,
        )
    except Exception as exc:
        print_manual_base_model_help(exc)
        raise RuntimeError("Automatic Stable Diffusion base model download failed.") from exc


def main() -> None:
    download_hypir_weights()
    download_base_model()


if __name__ == "__main__":
    main()
