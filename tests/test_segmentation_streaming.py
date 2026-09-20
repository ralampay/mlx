from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mlx.core.exceptions import MLXUserError
from mlx.modes.segmentation.streaming import (
    OpenCVSegmentationFrameSink,
    OpenCVSegmentationFrameSource,
)


class FakeCapture:
    def __init__(self, opened=True):
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        return False, None

    def release(self):
        self.released = True


@pytest.mark.parametrize("opened", [True, False])
def test_source_reuses_decoder_and_releases_failed_capture(monkeypatch, opened):
    capture = FakeCapture(opened)
    inputs = []

    def create_capture(value):
        inputs.append(value)
        return capture

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(VideoCapture=create_capture))
    if opened:
        source = OpenCVSegmentationFrameSource(source="camera", camera_index=3)
        assert source.capture is capture
        assert source.read() == (False, None)
        source.release()
        replacement = FakeCapture()
        source.capture = replacement
        source.release()
        assert replacement.released
    else:
        with pytest.raises(MLXUserError, match="Unable to open camera index 3"):
            OpenCVSegmentationFrameSource(source="camera", camera_index=3)
    assert inputs == [3]
    assert capture.released


def test_stream_ports_import_without_opencv():
    result = subprocess.run(
        [sys.executable, "-c", """
import sys
sys.modules['cv2'] = None
from mlx.modes.segmentation.streaming import OpenCVSegmentationFrameSource
from mlx.core.exceptions import MLXUserError
try:
    OpenCVSegmentationFrameSource(source='camera')
except MLXUserError as exc:
    assert 'Install opencv-python' in str(exc)
else:
    raise AssertionError('Missing OpenCV should fail at construction')
"""], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("key, keep_running", [(ord("q"), False), (27, False), (-1, True)])
def test_sink_preserves_stop_keys_and_window_settings(monkeypatch, key, keep_running):
    calls = []
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        imshow=lambda title, frame: calls.append(title),
        waitKey=lambda delay: calls.append(delay) or key,
        destroyAllWindows=lambda: calls.append("closed"),
    ))
    sink = OpenCVSegmentationFrameSink(title="segmentation", delay_ms=5)
    assert sink.show(np.zeros((2, 2, 3))) is keep_running
    sink.close()
    assert calls == ["segmentation", 5, "closed"]


@pytest.mark.parametrize("failure", ["load", "device", "report", "render", "release", None])
def test_command_cleans_up_ports_on_setup_and_loop_failures(monkeypatch, failure):
    from mlx.modes.segmentation import inference

    calls = []

    def step(name):
        calls.append(name)
        if failure == name:
            raise RuntimeError(name)

    model = SimpleNamespace(to=lambda device: step("device") or model, eval=lambda: None)

    def load(*args, **kwargs):
        step("load")
        return model, {"input_size": (2, 2)}

    monkeypatch.setattr(inference, "load_checkpoint_bundle", load)
    source = SimpleNamespace(
        read=lambda: (True, np.zeros((2, 2, 3))),
        release=lambda: step("release"),
    )
    sink = SimpleNamespace(show=lambda frame: False, close=lambda: step("close"))
    command = inference.RunSegmentationStreamInference(
        {}, "camera", frame_source=source, frame_sink=sink,
        reporter=SimpleNamespace(emit=lambda event: step("report")),
    )
    monkeypatch.setattr(command, "_render_frame", lambda frame: step("render") or frame)
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            command.execute()
    else:
        result = command.execute()
        assert result.frames_processed == 1
        assert result.stopped_by_user
    assert calls[-2:] == ["release", "close"]


@pytest.mark.parametrize("options, message", [
    ({"source": "video"}, "requires --file-path"),
    ({"source": "video", "file_path": "missing.mp4"}, "Video file not found"),
    ({"source": "unknown"}, "Unsupported stream source"),
])
def test_invalid_source_does_not_open_capture(monkeypatch, tmp_path, options, message):
    monkeypatch.chdir(tmp_path)

    def unexpected_capture(value):
        pytest.fail("Invalid input must be rejected before opening a capture")

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(VideoCapture=unexpected_capture))
    with pytest.raises(MLXUserError, match=message):
        OpenCVSegmentationFrameSource(**options)


def test_video_path_is_passed_to_shared_decoder(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.touch()
    capture = FakeCapture()
    inputs = []
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        VideoCapture=lambda value: inputs.append(value) or capture,
    ))
    source = OpenCVSegmentationFrameSource(source="video", file_path=str(video))
    assert inputs == [str(video)]
    source.release()
    assert capture.released
