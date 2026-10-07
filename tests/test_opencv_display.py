"""Portable contracts for OpenCV's Linux Qt display configuration."""

import sys
from types import SimpleNamespace

import pytest
import numpy as np

from mlx.core.opencv_display import configure_opencv_display
from mlx.core.streaming import OpenCVFrameSink


def test_missing_wheel_fonts_use_discovered_system_fonts(tmp_path):
    fonts = tmp_path / "fonts"
    fonts.mkdir()
    environ = {"QT_QPA_FONTDIR": str(tmp_path / "missing")}
    configure_opencv_display(environ, find_fonts=lambda: fonts, platform="linux")
    assert environ["QT_QPA_FONTDIR"] == str(fonts)


def test_explicit_fonts_survive_opencv_import_override(tmp_path, monkeypatch):
    fonts = tmp_path / "fonts"
    fonts.mkdir()
    environ = {"QT_QPA_FONTDIR": str(tmp_path / "wheel-missing")}
    configure_opencv_display(environ, font_directory=str(fonts), platform="linux")
    assert environ["QT_QPA_FONTDIR"] == str(fonts)


def test_font_lookup_is_optional_and_non_linux_configuration_is_preserved(tmp_path):
    original = {"QT_QPA_FONTDIR": str(tmp_path / "missing")}
    environ = dict(original)
    configure_opencv_display(environ, find_fonts=lambda: None, platform="linux")
    assert environ == original

    def fail_lookup():
        raise AssertionError("Non-Linux display must not probe fontconfig")

    configure_opencv_display(environ, find_fonts=fail_lookup, platform="darwin")
    assert environ == original


def test_wayland_session_uses_available_xcb_and_preserves_explicit_platform(tmp_path):
    platforms = tmp_path / "platforms"
    platforms.mkdir()
    (platforms / "libqxcb.so").touch()
    environ = {"QT_QPA_PLATFORM_PLUGIN_PATH": str(tmp_path), "XDG_SESSION_TYPE": "wayland", "DISPLAY": ":1"}
    configure_opencv_display(environ, platform="linux")
    assert environ["QT_QPA_PLATFORM"] == "xcb"
    environ["QT_QPA_PLATFORM"] = "offscreen"
    configure_opencv_display(environ, platform="linux")
    assert environ["QT_QPA_PLATFORM"] == "offscreen"


def test_wayland_plugin_and_absent_x11_display_are_respected(tmp_path):
    platforms = tmp_path / "platforms"
    platforms.mkdir()
    for name in ("libqxcb.so", "libqwayland-generic.so"):
        (platforms / name).touch()
    environ = {"QT_QPA_PLATFORM_PLUGIN_PATH": str(tmp_path), "XDG_SESSION_TYPE": "wayland", "DISPLAY": ":1"}
    configure_opencv_display(environ, platform="linux")
    assert "QT_QPA_PLATFORM" not in environ
    (platforms / "libqwayland-generic.so").unlink()
    del environ["DISPLAY"]
    configure_opencv_display(environ, platform="linux")
    assert "QT_QPA_PLATFORM" not in environ


def test_display_configuration_runs_before_first_window(monkeypatch):
    from mlx.core import opencv_display

    events = []
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        imshow=lambda *args: events.append("show"), waitKey=lambda *args: ord("q"),
        destroyAllWindows=lambda: events.append("close"),
    ))
    monkeypatch.setattr(opencv_display, "configure_opencv_display", lambda *args, **kwargs: events.append("configure"))
    sink = OpenCVFrameSink(title="test", delay_ms=1)
    assert not sink.show(None)
    sink.close()
    assert events == ["configure", "show", "close"]


@pytest.mark.parametrize("failure", [False, True])
def test_xwayland_session_hint_is_scoped_and_restored(failure):
    from mlx.core.opencv_display import opencv_display_environment

    environ = {"QT_QPA_PLATFORM": "xcb", "XDG_SESSION_TYPE": "wayland", "DISPLAY": ":1"}
    original = dict(environ)
    try:
        with opencv_display_environment(environ):
            assert environ["XDG_SESSION_TYPE"] == "x11"
            if failure:
                raise RuntimeError("window failed")
    except RuntimeError:
        assert failure
    assert environ == original


def test_native_wayland_session_hint_is_unchanged():
    from mlx.core.opencv_display import opencv_display_environment

    environ = {"QT_QPA_PLATFORM": "wayland", "XDG_SESSION_TYPE": "wayland"}
    with opencv_display_environment(environ):
        assert environ["XDG_SESSION_TYPE"] == "wayland"


@pytest.mark.parametrize("shape,expected", [
    ((720, 1280, 3), (540, 960, 3)),
    ((1920, 1080, 3), (540, 304, 3)),
    ((100, 200, 3), (100, 200, 3)),
])
def test_bounded_display_preserves_aspect_ratio_and_source_frame(monkeypatch, shape, expected):
    shown = []
    windows = []
    sizes = []

    def resize(frame, size, **kwargs):
        return np.zeros((size[1], size[0], 3), dtype=frame.dtype)

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        imshow=lambda title, frame: shown.append(frame), resize=resize, INTER_AREA=3,
        waitKey=lambda delay: -1,
        namedWindow=lambda title, flags: windows.append((title, flags)),
        resizeWindow=lambda title, width, height: sizes.append((title, width, height)),
        destroyAllWindows=lambda: None,
        WINDOW_NORMAL=0, WINDOW_KEEPRATIO=0, WINDOW_GUI_NORMAL=16,
    ))
    frame = np.zeros(shape, dtype=np.uint8)
    sink = OpenCVFrameSink(title="test", delay_ms=1, max_size=(960, 540))
    assert sink.show(frame)
    assert shown[0].shape == expected
    assert frame.shape == shape
    if shape == expected:
        assert shown[0] is frame
    assert windows == [("test", 16)]
    assert sizes == [("test", expected[1], expected[0])]
    assert sink.show(frame)
    assert len(windows) == len(sizes) == 1
    sink.close()
    assert sink.show(frame)
    assert len(windows) == len(sizes) == 2


@pytest.mark.parametrize("size", [(0, 540), (960, -1), (960,), (960, 540.0)])
def test_invalid_display_bounds_are_actionable(size):
    from mlx.core.exceptions import MLXUserError

    with pytest.raises(MLXUserError, match="positive integer"):
        OpenCVFrameSink(title="test", delay_ms=1, max_size=size)
