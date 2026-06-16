import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
FLASHVSR_CLI_PATH = ROOT / "scripts" / "flashvsr_cli.py"


def load_flashvsr_cli():
    spec = importlib.util.spec_from_file_location("cleanvideo_flashvsr_cli", FLASHVSR_CLI_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {FLASHVSR_CLI_PATH}")
    module = importlib.util.module_from_spec(spec)
    fake_imageio = types.SimpleNamespace(get_reader=None, get_writer=None)
    with patch.dict(sys.modules, {"imageio": fake_imageio}):
        spec.loader.exec_module(module)
    return module


class FakeSource:
    def __init__(self, *_args, **_kwargs) -> None:
        self.target_width = 16
        self.target_height = 16
        self.source_frames = 1
        self.model_frames = 25
        self.fps = 30.0
        self.discarded_before: list[int] = []
        self.closed = False

    def slice(self, start_index: int, end_index: int) -> torch.Tensor:
        frame_count = max(1, end_index - start_index)
        return torch.zeros((1, 3, frame_count, 16, 16), dtype=torch.float32)

    def discard_before(self, frame_index: int) -> None:
        self.discarded_before.append(frame_index)

    def close(self) -> None:
        self.closed = True


class FakeWriter:
    def __init__(self) -> None:
        self.frames = 0
        self.closed = False

    def append_data(self, _frame) -> None:
        self.frames += 1

    def close(self) -> None:
        self.closed = True


class FakeLqProj:
    def __init__(self) -> None:
        self.calls = 0

    def clear_cache(self) -> None:
        self.calls = 0

    def stream_forward(self, _video_clip):
        self.calls += 1
        if self.calls == 1:
            return None
        return [torch.zeros((1, 1, 1), dtype=torch.float32)]


class FakeDenoisingModel:
    def __init__(self) -> None:
        self.blocks = [object()]
        self.LQ_proj_in = FakeLqProj()


class FakeTcDecoder:
    def __init__(self) -> None:
        self.cleaned = False

    def clean_mem(self) -> None:
        self.cleaned = True

    def decode_video(self, *_args, **_kwargs) -> torch.Tensor:
        return torch.zeros((1, 1, 3, 16, 16), dtype=torch.float32)


class FakePipe:
    def __init__(self) -> None:
        self.device = "cpu"
        self.torch_dtype = torch.float32
        self.dit = FakeDenoisingModel()
        self.TCDecoder = FakeTcDecoder()
        self.timestep = torch.tensor([1000.0])
        self.t_mod = torch.zeros((1, 6, 1), dtype=torch.float32)
        self.t = torch.zeros((1, 1), dtype=torch.float32)
        self.load_calls: list[list[str]] = []

    def check_resize_height_width(self, height: int, width: int) -> tuple[int, int]:
        return height, width

    def denoising_model(self) -> FakeDenoisingModel:
        return self.dit

    def load_models_to_device(self, names=None) -> None:
        self.load_calls.append(list(names or []))

    def ColorCorrector(self, frames, *_args, **_kwargs):
        return frames


class FakePipelineModule:
    @staticmethod
    def model_fn_wan_video(_dit, *, x, pre_cache_k, pre_cache_v, **_kwargs):
        return torch.zeros_like(x), pre_cache_k, pre_cache_v


class FlashVsrCliTests(unittest.TestCase):
    def test_streaming_render_uses_wrapped_modules_without_full_dit_shuttling(self) -> None:
        flashvsr_cli = load_flashvsr_cli()
        pipe = FakePipe()
        writer = FakeWriter()

        with TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(
                input=Path(temp_dir) / "input.mp4",
                output=Path(temp_dir) / "output.mp4",
                scale=2.0,
                total_frames=1,
                fps=30.0,
                seed=231,
                sparse_ratio=2.0,
                local_range=11,
                quality=6,
                live_original=None,
                live_enhanced=None,
                live_quality=92,
            )

            with (
                patch.object(flashvsr_cli, "StreamingLqFrameBuffer", FakeSource),
                patch.object(flashvsr_cli, "pipeline_module", return_value=FakePipelineModule),
                patch.object(flashvsr_cli.imageio, "get_writer", return_value=writer),
            ):
                flashvsr_cli.stream_flashvsr_tiny(object(), pipe, args)

        self.assertEqual(pipe.load_calls, [])
        self.assertEqual(writer.frames, 1)
        self.assertTrue(writer.closed)


if __name__ == "__main__":
    unittest.main()
