from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app import main


class WorkCleanupTests(unittest.TestCase):
    def setUp(self) -> None:
        with main.preview_lock:
            main.active_preview_tokens.clear()
            main.preview_processes.clear()
            main.cancelled_previews.clear()
            main.cleanup_in_progress = False

    def test_clear_work_dir_removes_uploads_adapters_and_unknown_entries(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "work"
            work_dirs = [
                root / "uploads",
                root / "previews",
                root / "exports",
                root / "jobs",
                root / "cache",
                root / "partials",
                root / "adapters",
            ]
            patches = [
                patch.object(main, "WORK_DIR", root),
                patch.object(main, "UPLOAD_DIR", root / "uploads"),
                patch.object(main, "PREVIEW_DIR", root / "previews"),
                patch.object(main, "EXPORT_DIR", root / "exports"),
                patch.object(main, "JOB_DIR", root / "jobs"),
                patch.object(main, "CACHE_DIR", root / "cache"),
                patch.object(main, "PARTIAL_DIR", root / "partials"),
                patch.object(main, "ADAPTER_DIR", root / "adapters"),
                patch.object(main, "WORK_DIRS", work_dirs),
            ]

            for active_patch in patches:
                active_patch.start()
            try:
                main.ensure_work_directories()
                for path in [
                    root / "uploads" / "source.mp4",
                    root / "uploads" / "video.json",
                    root / "exports" / "render.mp4",
                    root / "cache" / "cache-key" / "enhanced" / "frame_000001.png",
                    root / "adapters" / "adapter-1" / "adapter.json",
                    root / "adapters" / "adapter-1" / "state_dict.pth",
                    root / "unexpected" / "large.bin",
                    root / "loose-file.tmp",
                ]:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"x" * 16)

                result = main.clear_work_dir()

                self.assertFalse((root / "uploads" / "source.mp4").exists())
                self.assertFalse((root / "adapters" / "adapter-1").exists())
                self.assertFalse((root / "unexpected").exists())
                self.assertFalse((root / "loose-file.tmp").exists())
                for directory in work_dirs:
                    self.assertTrue(directory.is_dir())
                    self.assertEqual(list(directory.iterdir()), [])
                self.assertGreaterEqual(result["filesDeleted"], 8)
                self.assertGreater(result["bytesFreed"], 0)
                self.assertTrue(result["workCleared"])
                self.assertFalse(result["uploadsPreserved"])
                self.assertFalse(result["adaptersPreserved"])
            finally:
                for active_patch in reversed(patches):
                    active_patch.stop()

    def test_cleanup_endpoint_refuses_to_delete_while_preview_is_active(self) -> None:
        token = main.start_preview_run()
        try:
            with patch.object(main, "clear_work_dir") as clear_work_dir:
                with self.assertRaises(HTTPException) as raised:
                    main.cleanup_generated()

            self.assertEqual(raised.exception.status_code, 409)
            self.assertIn("preview request", str(raised.exception.detail))
            clear_work_dir.assert_not_called()
        finally:
            main.finish_preview_run(token)

    def test_preview_cannot_start_while_cleanup_window_is_active(self) -> None:
        main.begin_cleanup_window()
        try:
            with self.assertRaisesRegex(RuntimeError, "Cleanup is in progress"):
                main.start_preview_run()
        finally:
            main.finish_cleanup_window()


if __name__ == "__main__":
    unittest.main()
