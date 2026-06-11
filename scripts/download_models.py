from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    hf_hub_download(
        repo_id="lxq007/HYPIR",
        filename="HYPIR_sd2.pth",
        local_dir=ROOT / "models" / "hypir",
    )
    snapshot_download(
        repo_id="Manojb/stable-diffusion-2-1-base",
        local_dir=ROOT / "models" / "stable-diffusion-2-1-base",
        allow_patterns=[
            "scheduler/*",
            "tokenizer/*",
            "text_encoder/*",
            "unet/*",
            "vae/*",
            "model_index.json",
        ],
    )


if __name__ == "__main__":
    main()

