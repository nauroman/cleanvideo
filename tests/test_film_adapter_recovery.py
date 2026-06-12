from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.film_adapter import (
    cleanup_adapter_training_artifacts,
    set_train_config_resume_checkpoint,
    write_train_config,
)


class FilmAdapterRecoveryTests(unittest.TestCase):
    def test_resume_checkpoint_updates_generated_train_config(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "hypir_train.yaml"
            output_dir = root / "training"
            checkpoint_dir = output_dir / "checkpoint-1120"

            write_train_config(
                config_path=config_path,
                output_dir=output_dir,
                parquet_path=root / "dataset.parquet",
                base_model_path=root / "base-model",
                max_train_steps=1200,
                checkpointing_steps=400,
                checkpoints_total_limit=3,
            )

            set_train_config_resume_checkpoint(config_path, checkpoint_dir)
            text = config_path.read_text(encoding="utf-8")
            self.assertIn(f"resume_from_checkpoint: {checkpoint_dir.as_posix()}", text)
            self.assertNotIn("resume_from_checkpoint: ~", text)

            set_train_config_resume_checkpoint(config_path, None)
            text = config_path.read_text(encoding="utf-8")
            self.assertIn("resume_from_checkpoint: ~", text)

    def test_cleanup_removes_training_artifacts_but_keeps_final_weights(self) -> None:
        with TemporaryDirectory() as temp_dir:
            adapter_root = Path(temp_dir)
            final_checkpoint = adapter_root / "training" / "checkpoint-1200"
            old_checkpoint = adapter_root / "training" / "checkpoint-800"
            final_checkpoint.mkdir(parents=True)
            old_checkpoint.mkdir(parents=True)

            for path in [
                adapter_root / "source_frames" / "sample_000001.png",
                adapter_root / "patches" / "patch_000001_01.png",
                adapter_root / "dataset.parquet",
                adapter_root / "dataset.json",
                adapter_root / "hypir_train.yaml",
                adapter_root / "train.log",
                adapter_root / "training" / "logs" / "events.out.tfevents",
                old_checkpoint / "state_dict.pth",
                final_checkpoint / "state_dict.pth",
                final_checkpoint / "optimizer.bin",
                final_checkpoint / "ema_state_dict.pth",
            ]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")

            cleanup_adapter_training_artifacts(adapter_root, final_checkpoint)

            self.assertTrue((final_checkpoint / "state_dict.pth").exists())
            self.assertFalse((final_checkpoint / "optimizer.bin").exists())
            self.assertFalse((final_checkpoint / "ema_state_dict.pth").exists())
            self.assertFalse(old_checkpoint.exists())
            self.assertFalse((adapter_root / "source_frames").exists())
            self.assertFalse((adapter_root / "patches").exists())
            self.assertFalse((adapter_root / "dataset.parquet").exists())
            self.assertFalse((adapter_root / "dataset.json").exists())
            self.assertFalse((adapter_root / "hypir_train.yaml").exists())
            self.assertFalse((adapter_root / "train.log").exists())
            self.assertFalse((adapter_root / "training" / "logs").exists())


if __name__ == "__main__":
    unittest.main()
