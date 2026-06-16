import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.flashvsr_engine import FlashVsrEngine, FlashVsrSettings, is_transient_import_corruption


class FakeFlashVsrProcess:
    def __init__(self, lines: list[str], return_code: int, output_path: Path | None = None) -> None:
        self.stdout = iter(f"{line}\n" for line in lines)
        self.return_code = return_code
        self.output_path = output_path

    def poll(self) -> int:
        return self.return_code

    def wait(self, timeout: float | None = None) -> int:
        if self.return_code == 0 and self.output_path is not None:
            self.output_path.write_bytes(b"ok")
        return self.return_code

    def terminate(self) -> None:
        return None

    def kill(self) -> None:
        return None


class FlashVsrEngineTests(unittest.TestCase):
    def test_detects_python_re_parser_tokenizer_corruption(self) -> None:
        detail = """
        File "/home/user/.cleanvideo/flashvsr-wsl/.venv/lib/python3.11/site-packages/urllib3/util/url.py", line 59, in <module>
          _IPV6_RE = re.compile("^" + _IPV6_PAT + "$")
        File "/home/user/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/lib/python3.11/re/_parser.py", line 664, in _parse
          hi += sourceget()
        TypeError: unsupported operand type(s) for +: 'int' and 'Tokenizer'
        """

        self.assertTrue(is_transient_import_corruption(detail))

    def test_does_not_mask_regular_flashvsr_failures(self) -> None:
        detail = "RuntimeError: CUDA out of memory while running FlashVSR"

        self.assertFalse(is_transient_import_corruption(detail))

    def test_retries_transient_import_corruption_once(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.mp4"
            output_path = root / "output.mp4"
            input_path.write_bytes(b"input")
            attempts = 0
            lines: list[str] = []

            def fake_popen(*_args, **_kwargs):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    return FakeFlashVsrProcess(
                        [
                            'File "/home/user/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/lib/python3.11/re/_parser.py", line 664, in _parse',
                            "TypeError: unsupported operand type(s) for +: 'int' and 'Tokenizer'",
                        ],
                        1,
                    )
                return FakeFlashVsrProcess(["Done"], 0, output_path)

            engine = FlashVsrEngine()
            engine._active_backend = lambda: "windows"  # type: ignore[method-assign]

            with (
                patch("app.flashvsr_engine.subprocess.Popen", side_effect=fake_popen),
                patch("app.flashvsr_engine.time.sleep", return_value=None),
            ):
                result = engine.enhance_video(
                    input_path,
                    output_path,
                    FlashVsrSettings(),
                    on_line=lines.append,
                )

            self.assertEqual(result["seed"], 231)
            self.assertEqual(attempts, 2)
            self.assertTrue(output_path.exists())
            self.assertTrue(any("transient Python import corruption" in line for line in lines))

    def test_streaming_export_passes_frame_metadata_to_cli(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.mp4"
            output_path = root / "output.mp4"
            input_path.write_bytes(b"input")
            captured_args: list[str] = []

            def fake_popen(args, *_args, **_kwargs):
                captured_args.extend(str(arg) for arg in args)
                return FakeFlashVsrProcess(["Done"], 0, output_path)

            engine = FlashVsrEngine()
            engine._active_backend = lambda: "windows"  # type: ignore[method-assign]

            with patch("app.flashvsr_engine.subprocess.Popen", side_effect=fake_popen):
                engine.enhance_video(
                    input_path,
                    output_path,
                    FlashVsrSettings(variant="tiny_long"),
                    total_frames=123,
                    fps=23.976,
                    streaming=True,
                )

            self.assertIn("--streaming", captured_args)
            self.assertIn("--total_frames", captured_args)
            self.assertIn("123", captured_args)
            self.assertIn("--fps", captured_args)
            self.assertIn("23.976", captured_args)


if __name__ == "__main__":
    unittest.main()
