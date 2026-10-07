"""Configure the Linux Qt bundle used by OpenCV display windows."""

from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys


def system_font_directory():
    """Ask fontconfig for an installed font, if fontconfig is available."""
    try:
        result = subprocess.run(
            ["fc-match", "--format=%{file}"], capture_output=True, text=True,
            check=False, timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None  # Fontconfig is an optional system facility.
    path = Path(result.stdout.strip())
    return path.parent if result.returncode == 0 and path.is_file() else None


def configure_opencv_display(environ, *, font_directory=None,
                             find_fonts=system_font_directory, platform=sys.platform):
    """Fix missing bundled fonts and select only an available Qt platform.

    Call after importing cv2: wheel initialization overwrites QT_QPA_FONTDIR.
    Existing usable font and explicit platform selections take precedence.
    """
    if not platform.startswith("linux"):
        return
    configured_fonts = font_directory or environ.get("QT_QPA_FONTDIR")
    if configured_fonts and Path(configured_fonts).is_dir():
        environ["QT_QPA_FONTDIR"] = str(configured_fonts)
    elif "QT_QPA_FONTDIR" in environ:
        installed_fonts = find_fonts()
        if installed_fonts is not None:
            environ["QT_QPA_FONTDIR"] = str(installed_fonts)

    if (environ.get("QT_QPA_PLATFORM") or environ.get("XDG_SESSION_TYPE") != "wayland"
            or not environ.get("DISPLAY")):
        return
    plugin_root = environ.get("QT_QPA_PLATFORM_PLUGIN_PATH")
    if not plugin_root:
        return
    plugins = Path(plugin_root) / "platforms"
    if (plugins / "libqxcb.so").is_file() and not any(plugins.glob("*wayland*.so")):
        environ["QT_QPA_PLATFORM"] = "xcb"


@contextmanager
def opencv_display_environment(environ):
    """Give Qt an X11 session hint while opening an XWayland window."""
    session = environ.get("XDG_SESSION_TYPE")
    use_x11 = (session == "wayland" and environ.get("QT_QPA_PLATFORM") == "xcb"
               and bool(environ.get("DISPLAY")))
    if use_x11:
        environ["XDG_SESSION_TYPE"] = "x11"
    try:
        yield
    finally:
        if use_x11:
            environ["XDG_SESSION_TYPE"] = session
