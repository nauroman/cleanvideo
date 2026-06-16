from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from app.main import flashvsr_chunk_frames_done_from_line, reusable_video_file


class FlashVsrResumeTests(unittest.TestCase):
    def test_flashvsr_frame_progress_lines_report_chunk_frames_done(self) -> None:
        self.assertEqual(flashvsr_chunk_frames_done_from_line("0 25", 21), 1)
        self.assertEqual(flashvsr_chunk_frames_done_from_line("20 25", 21), 21)
        self.assertEqual(flashvsr_chunk_frames_done_from_line("20 25", 5), 5)
        self.assertIsNone(flashvsr_chunk_frames_done_from_line("25 25", 21))
        self.assertIsNone(flashvsr_chunk_frames_done_from_line("0 NVIDIA GeForce RTX 4090", 21))
        self.assertIsNone(flashvsr_chunk_frames_done_from_line("Saving chunk_00001.mp4: 5/21", 21))

    def test_reusable_video_file_requires_existing_nonempty_valid_video(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "chunk.mp4"

            self.assertFalse(reusable_video_file(path, 21))

            path.write_bytes(b"not a real mp4")
            with patch("app.main.probe_video", side_effect=RuntimeError("bad video")):
                self.assertFalse(reusable_video_file(path, 21))

    def test_reusable_video_file_rejects_short_chunks(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "chunk.mp4"
            path.write_bytes(b"video")

            with patch("app.main.probe_video", return_value={"frameCount": 10}):
                self.assertFalse(reusable_video_file(path, 21))

            with patch("app.main.probe_video", return_value={"frameCount": 21}):
                self.assertTrue(reusable_video_file(path, 21))


if __name__ == "__main__":
    unittest.main()
