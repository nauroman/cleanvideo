from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.dove_engine import DoveEngine, DoveSettings
from app.main import ENGINE_VERSION, ProcessSettings
from app.supir_engine import SupirEngine, SupirSettings


class DoveEngineTests(unittest.TestCase):
    def test_status_reports_required_dove_layout(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            engine = DoveEngine(root=root)

            status = engine.status()

            self.assertFalse(status["available"])
            self.assertEqual(status["repoPath"], str(root / "external" / "DOVE"))
            self.assertEqual(status["pythonPath"], str(root / ".venv-dove" / "Scripts" / "python.exe"))
            self.assertEqual(status["modelPath"], str(root / "external" / "DOVE" / "pretrained_models" / "DOVE"))
            self.assertIn(str(root / "external" / "DOVE" / "inference_script.py"), status["missing"])

    def test_build_command_uses_isolated_input_and_output_dirs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input" / "clip.mp4"
            output_path = root / "output" / "result.mp4"
            work_dir = root / "work"
            engine = DoveEngine(root=root)

            command = engine.build_command(
                input_path,
                output_path,
                DoveSettings(upscale=2, seed=123, chunk_len=17, overlap_t=4),
                work_dir=work_dir,
                fps=24.0,
            )

            self.assertEqual(command[0], str(root / ".venv-dove" / "Scripts" / "python.exe"))
            self.assertIn(str(root / "external" / "DOVE" / "inference_script.py"), command)
            self.assertIn("--input_dir", command)
            self.assertIn(str(work_dir / "input"), command)
            self.assertIn("--output_path", command)
            self.assertIn(str(work_dir / "output"), command)
            self.assertIn("--model_path", command)
            self.assertIn(str(root / "external" / "DOVE" / "pretrained_models" / "DOVE"), command)
            self.assertIn("--is_vae_st", command)
            self.assertIn("--save_format", command)
            self.assertIn("yuv420p", command)

    def test_status_checks_required_dove_model_subfolders(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = root / "external" / "DOVE"
            model = repo / "pretrained_models" / "DOVE"
            (root / ".venv-dove" / "Scripts").mkdir(parents=True)
            (root / ".venv-dove" / "Scripts" / "python.exe").touch()
            repo.mkdir(parents=True)
            (repo / "inference_script.py").touch()
            model.mkdir(parents=True)
            (model / "model_index.json").touch()
            for folder in ["scheduler", "text_encoder", "tokenizer", "transformer"]:
                (model / folder).mkdir()

            status = DoveEngine(root=root).status()

            self.assertFalse(status["available"])
            self.assertIn(str(model / "vae"), status["missing"])

    def test_build_command_skips_chunk_args_for_short_clip(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            command = DoveEngine(root=root).build_command(
                root / "input.mp4",
                root / "out.mp4",
                DoveSettings(upscale=1, chunk_len=9, overlap_t=4),
                work_dir=root / "work",
                frame_count=1,
            )

            self.assertNotIn("--chunk_len", command)
            self.assertNotIn("--overlap_t", command)


class SupirEngineTests(unittest.TestCase):
    def test_status_reports_required_supir_layout(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            engine = SupirEngine(root=root)

            status = engine.status()

            self.assertFalse(status["available"])
            self.assertEqual(status["repoPath"], str(root / "external" / "SUPIR"))
            self.assertEqual(status["pythonPath"], str(root / ".venv-supir" / "Scripts" / "python.exe"))
            self.assertEqual(status["configPath"], str(root / "external" / "SUPIR" / "options" / "SUPIR_v0.yaml"))
            self.assertIn(str(root / "external" / "SUPIR" / "test.py"), status["missing"])

    def test_build_command_uses_supir_folder_cli(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "frames"
            output_dir = root / "enhanced"
            engine = SupirEngine(root=root)

            command = engine.build_command(
                input_dir,
                output_dir,
                SupirSettings(upscale=2, seed=123, sign="F", no_llava=True, color_fix_type="AdaIn"),
            )

            self.assertEqual(command[0], str(root / ".venv-supir" / "Scripts" / "python.exe"))
            self.assertIn(str(root / "external" / "SUPIR" / "test.py"), command)
            self.assertIn("--img_dir", command)
            self.assertIn(str(input_dir), command)
            self.assertIn("--save_dir", command)
            self.assertIn(str(output_dir), command)
            self.assertIn("--SUPIR_sign", command)
            self.assertIn("F", command)
            self.assertIn("--no_llava", command)
            self.assertEqual(engine.output_for_input(output_dir, Path("000123.png")), output_dir / "000123_0.png")

    def test_status_checks_supir_clip_model_paths(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = root / "external" / "SUPIR"
            config = repo / "options" / "SUPIR_v0.yaml"
            ckpt_config = repo / "CKPT_PTH.py"
            (root / ".venv-supir" / "Scripts").mkdir(parents=True)
            (root / ".venv-supir" / "Scripts" / "python.exe").touch()
            repo.mkdir(parents=True)
            (repo / "test.py").touch()
            config.parent.mkdir(parents=True)
            config.write_text(
                "\n".join(
                    [
                        f"SDXL_CKPT: {root / 'sdxl.safetensors'}",
                        f"SUPIR_CKPT_Q: {root / 'supir-q.ckpt'}",
                    ]
                ),
                encoding="utf-8",
            )
            (root / "sdxl.safetensors").touch()
            (root / "supir-q.ckpt").touch()
            ckpt_config.write_text(
                "\n".join(
                    [
                        f"SDXL_CLIP1_PATH = r'{root / 'clip1'}'",
                        f"SDXL_CLIP2_CKPT_PTH = r'{root / 'clip2.bin'}'",
                    ]
                ),
                encoding="utf-8",
            )

            status = SupirEngine(root=root).status()

            self.assertFalse(status["available"])
            self.assertIn(f"SDXL_CLIP1_PATH: {root / 'clip1'}", status["missing"])
            self.assertIn(f"SDXL_CLIP2_CKPT_PTH: {root / 'clip2.bin'}", status["missing"])


class EngineRegistryTests(unittest.TestCase):
    def test_process_settings_accepts_new_engines(self) -> None:
        self.assertIn("dove", ENGINE_VERSION)
        self.assertIn("supir", ENGINE_VERSION)
        dove_settings = ProcessSettings(
            engine="dove",
            doveCpuOffload=True,
            doveChunkLength=9,
            doveTemporalOverlap=64,
        ).to_dove()
        supir_settings = ProcessSettings(engine="supir", supirSteps=4, supirMinSize=256).to_supir()
        self.assertEqual(dove_settings.upscale, 1)
        self.assertTrue(dove_settings.cpu_offload)
        self.assertEqual(dove_settings.chunk_len, 9)
        self.assertEqual(dove_settings.overlap_t, 8)
        self.assertEqual(supir_settings.upscale, 1)
        self.assertEqual(supir_settings.edm_steps, 4)
        self.assertEqual(supir_settings.min_size, 256)


if __name__ == "__main__":
    unittest.main()
