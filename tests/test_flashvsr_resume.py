from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app.main import (
    ExportRequest,
    JobState,
    PartialExportRequest,
    export_partial_video,
    flashvsr_native_export_blocker,
    flashvsr_progress_frames_done_from_line,
    jobs,
    reusable_video_file,
)


class FlashVsrResumeTests(unittest.TestCase):
    def test_flashvsr_frame_progress_lines_report_streaming_frames_done(self) -> None:
        self.assertEqual(flashvsr_progress_frames_done_from_line("0 25", 21), 1)
        self.assertEqual(flashvsr_progress_frames_done_from_line("20 25", 21), 21)
        self.assertEqual(flashvsr_progress_frames_done_from_line("20 25", 5), 5)
        self.assertIsNone(flashvsr_progress_frames_done_from_line("25 25", 21))
        self.assertIsNone(flashvsr_progress_frames_done_from_line("0 NVIDIA GeForce RTX 4090", 21))
        self.assertIsNone(flashvsr_progress_frames_done_from_line("Saving chunk_00001.mp4: 5/21", 21))

    def test_flashvsr_full_render_is_blocked_because_it_requires_chunking(self) -> None:
        request = ExportRequest(
            videoId="video-1",
            engine="flashvsr",
            upscale=2,
            flashvsrVariant="full",
        )
        blocker = flashvsr_native_export_blocker(
            {"width": 852, "height": 480, "frameCount": 1612},
            request,
        )

        self.assertIsNotNone(blocker)
        self.assertIn("continuous streaming", blocker or "")

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

    def test_flashvsr_partial_export_rejects_streaming_render_without_chunks(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            job_id = "flashvsr-partial-test"
            stamp = "2026-06-16T12:00:00"
            jobs[job_id] = JobState(
                id=job_id,
                kind="export",
                status="running",
                progress=0.5,
                message="FlashVSR completed chunk 1/2",
                engine="flashvsr",
                videoId="video-1",
                framesDone=21,
                framesTotal=42,
                partialFramesReady=21,
                cacheKey="cache-key",
                startedAt=stamp,
                updatedAt=stamp,
            )

            try:
                with patch(
                    "app.main.get_video",
                    return_value={
                        "name": "source.mp4",
                        "path": str(source),
                        "metadata": {"fps": 30.0, "frameCount": 42},
                    },
                ):
                    with self.assertRaises(HTTPException) as ctx:
                        export_partial_video(job_id, PartialExportRequest())
            finally:
                jobs.pop(job_id, None)

            self.assertEqual(ctx.exception.status_code, 400)
            self.assertIn("render chunks are disabled", str(ctx.exception.detail))


if __name__ == "__main__":
    unittest.main()
