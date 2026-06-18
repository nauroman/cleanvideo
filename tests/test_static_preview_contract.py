from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "static" / "app.js"


class StaticPreviewContractTests(unittest.TestCase):
    def test_preview_timeout_is_capped_at_two_minutes(self) -> None:
        script = APP_JS.read_text(encoding="utf-8")

        self.assertIn("const PREVIEW_TIMEOUT_MS = 2 * 60 * 1000;", script)
        self.assertIn("return PREVIEW_TIMEOUT_MS;", script)
        self.assertNotIn("previewMaxTimeoutMsByEngine", script)
        self.assertNotIn("30 * 60 * 1000", script)

    def test_preview_timeout_message_suggests_lower_settings(self) -> None:
        script = APP_JS.read_text(encoding="utf-8")

        self.assertIn("function previewTimeoutAdvice", script)
        self.assertIn("Resolution", script)
        self.assertIn("Steps", script)
        self.assertIn("Min Size", script)
        self.assertIn("Chunk Length", script)
        self.assertIn("Preview timed out after", script)


if __name__ == "__main__":
    unittest.main()
