from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from app.main import (
    FLASHVSR_EXPORT_CHUNK_FRAMES,
    JobState,
    PartialExportRequest,
    contiguous_flashvsr_chunks,
    export_partial_video,
    flashvsr_chunk_frames_done_from_line,
    jobs,
    reusable_video_file,
)


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

    def test_contiguous_flashvsr_chunks_stop_at_first_missing_chunk(self) -> None:
        with TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir)
            checked: list[tuple[str, int]] = []

            def fake_reusable(path: Path, min_frames: int | None = None) -> bool:
                checked.append((path.name, int(min_frames or 0)))
                return path.name in {"chunk_00001.mp4", "chunk_00002.mp4"}

            with patch("app.main.reusable_video_file", side_effect=fake_reusable):
                chunks, frames_ready = contiguous_flashvsr_chunks(cache_root, 50)

            self.assertEqual([path.name for path in chunks], ["chunk_00001.mp4", "chunk_00002.mp4"])
            self.assertEqual(frames_ready, FLASHVSR_EXPORT_CHUNK_FRAMES * 2)
            self.assertEqual(
                checked,
                [
                    ("chunk_00001.mp4", FLASHVSR_EXPORT_CHUNK_FRAMES),
                    ("chunk_00002.mp4", FLASHVSR_EXPORT_CHUNK_FRAMES),
                    ("chunk_00003.mp4", 50 - FLASHVSR_EXPORT_CHUNK_FRAMES * 2),
                ],
            )

    def test_flashvsr_partial_export_concats_completed_chunks(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            output = root / "partial.mp4"
            chunks = [root / "chunk_00001.mp4"]
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
                with (
                    patch(
                        "app.main.get_video",
                        return_value={
                            "name": "source.mp4",
                            "path": str(source),
                            "metadata": {"fps": 30.0, "frameCount": 42},
                        },
                    ),
                    patch("app.main.contiguous_flashvsr_chunks", return_value=(chunks, 21)) as ready_chunks,
                    patch("app.main.concat_videos_copy") as concat,
                    patch("app.main.remux_video_with_source_audio", return_value="video copy, audio copy") as remux,
                    patch("app.main.partial_output_path", return_value=output),
                    patch("app.main.encode_video") as encode,
                ):
                    result = export_partial_video(job_id, PartialExportRequest())
            finally:
                jobs.pop(job_id, None)

            ready_chunks.assert_called_once()
            concat.assert_called_once()
            remux.assert_called_once()
            encode.assert_not_called()
            self.assertEqual(result["framesDone"], 21)
            self.assertEqual(result["framesTotal"], 42)
            self.assertEqual(result["durationSeconds"], 0.7)
            self.assertEqual(result["encoder"], "video copy, audio copy")


if __name__ == "__main__":
    unittest.main()
