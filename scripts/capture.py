"""Screenshot of one window or one screen (Windows).

    python scripts/capture.py --list
    python scripts/capture.py --screen 1
    python scripts/capture.py --window "Label Studio" --delay 3
    python scripts/capture.py --window Chrome -o captures/manuel/chrome.png

``--screen 0`` captures the whole virtual desktop (all monitors). Minimised windows
cannot be captured. A window is rendered by Windows (``PrintWindow``), so covering
windows do not appear in its picture; if that call fails, the screen pixels are used
instead and the window must be visible (``--delay`` helps).
Default output: ``captures/manuel/AAAAMMJJ_HHMMSS.png`` (outside the watched folder).
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from collections.abc import Sequence
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401 - makes ``src`` importable
from PIL import Image, ImageGrab

from src.config.loader import PROJECT_ROOT

Bbox = tuple[int, int, int, int]  # left, top, right, bottom

DWMWA_EXTENDED_FRAME_BOUNDS = 9
MONITORINFOF_PRIMARY = 0x1
PW_RENDERFULLCONTENT = 0x2
BI_RGB = 0
DIB_RGB_COLORS = 0
SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 76, 77, 78, 79
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "captures" / "manuel"


@dataclass(frozen=True)
class WindowInfo:
    handle: int
    title: str
    bbox: Bbox  # visible frame, in screen coordinates
    outer: Bbox  # full window rectangle, including the invisible resize border

    @property
    def size(self) -> tuple[int, int]:
        return self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]

    @property
    def outer_size(self) -> tuple[int, int]:
        return self.outer[2] - self.outer[0], self.outer[3] - self.outer[1]


@dataclass(frozen=True)
class ScreenInfo:
    index: int  # 0 = whole virtual desktop, then 1, 2, ... for each monitor
    bbox: Bbox
    primary: bool

    @property
    def size(self) -> tuple[int, int]:
        return self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]


if sys.platform == "win32":
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _gdi32 = ctypes.WinDLL("gdi32")
    _dwmapi = ctypes.WinDLL("dwmapi")

    class _MonitorInfo(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", wintypes.RECT),
            ("rcWork", wintypes.RECT),
            ("dwFlags", wintypes.DWORD),
        ]

    class _BitmapInfoHeader(ctypes.Structure):
        _fields_ = [
            ("biSize", wintypes.DWORD),
            ("biWidth", wintypes.LONG),
            ("biHeight", wintypes.LONG),
            ("biPlanes", wintypes.WORD),
            ("biBitCount", wintypes.WORD),
            ("biCompression", wintypes.DWORD),
            ("biSizeImage", wintypes.DWORD),
            ("biXPelsPerMeter", wintypes.LONG),
            ("biYPelsPerMeter", wintypes.LONG),
            ("biClrUsed", wintypes.DWORD),
            ("biClrImportant", wintypes.DWORD),
        ]

    _WindowEnumProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    _MonitorEnumProc = ctypes.WINFUNCTYPE(
        wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.LPARAM
    )
    _user32.EnumWindows.argtypes = [_WindowEnumProc, wintypes.LPARAM]
    _user32.EnumDisplayMonitors.argtypes = [wintypes.HDC, ctypes.c_void_p, _MonitorEnumProc, wintypes.LPARAM]
    _user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.POINTER(_MonitorInfo)]
    _user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    _user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    _user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    _user32.GetDC.argtypes = [wintypes.HWND]
    _user32.GetDC.restype = wintypes.HDC
    _user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    _user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
    _gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    _gdi32.CreateCompatibleDC.restype = wintypes.HDC
    _gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
    _gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
    _gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    _gdi32.SelectObject.restype = wintypes.HGDIOBJ
    _gdi32.GetDIBits.argtypes = [
        wintypes.HDC,
        wintypes.HBITMAP,
        wintypes.UINT,
        wintypes.UINT,
        ctypes.c_void_p,
        ctypes.POINTER(_BitmapInfoHeader),
        wintypes.UINT,
    ]
    _gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    _gdi32.DeleteDC.argtypes = [wintypes.HDC]


def _enable_dpi_awareness() -> None:
    """Without it, Windows scales window rectangles and they stop matching the screen pixels."""
    try:
        ctypes.WinDLL("shcore").SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except OSError:
        pass  # Windows without shcore: nothing to do


def _rect_to_bbox(rect: wintypes.RECT) -> Bbox:
    return rect.left, rect.top, rect.right, rect.bottom


def _window_title(handle: int) -> str:
    length = _user32.GetWindowTextLengthW(handle)
    buffer = ctypes.create_unicode_buffer(length + 1)
    _user32.GetWindowTextW(handle, buffer, length + 1)
    return buffer.value


def _window_bounds(handle: int) -> tuple[Bbox, Bbox]:
    """(visible frame, full rectangle). The full one includes the invisible border of Windows 10/11."""
    outer = wintypes.RECT()
    _user32.GetWindowRect(handle, ctypes.byref(outer))
    visible = wintypes.RECT()
    hresult = _dwmapi.DwmGetWindowAttribute(
        handle, DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(visible), ctypes.sizeof(visible)
    )
    if hresult != 0:
        visible = outer
    return _rect_to_bbox(visible), _rect_to_bbox(outer)


def list_windows() -> list[WindowInfo]:
    """Visible, non-minimised windows with a title, front-most first."""
    windows: list[WindowInfo] = []

    def visit(handle: int, _lparam: int) -> bool:
        if _user32.IsWindowVisible(handle) and not _user32.IsIconic(handle):
            title = _window_title(handle)
            visible, outer = _window_bounds(handle)
            info = WindowInfo(handle, title, visible, outer)
            if title.strip() and all(size > 0 for size in info.size):
                windows.append(info)
        return True

    _user32.EnumWindows(_WindowEnumProc(visit), 0)
    return windows


def list_screens() -> list[ScreenInfo]:
    """Index 0 is the whole virtual desktop, then each monitor in enumeration order."""
    left = _user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    top = _user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    width = _user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
    height = _user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
    screens = [ScreenInfo(0, (left, top, left + width, top + height), primary=False)]

    def visit(monitor: int, _hdc: int, _rect: object, _lparam: int) -> bool:
        info = _MonitorInfo()
        info.cbSize = ctypes.sizeof(_MonitorInfo)
        _user32.GetMonitorInfoW(monitor, ctypes.byref(info))
        primary = bool(info.dwFlags & MONITORINFOF_PRIMARY)
        screens.append(ScreenInfo(len(screens), _rect_to_bbox(info.rcMonitor), primary))
        return True

    _user32.EnumDisplayMonitors(None, None, _MonitorEnumProc(visit), 0)
    return screens


def find_windows(query: str, windows: Sequence[WindowInfo]) -> list[WindowInfo]:
    """Windows whose title contains ``query`` (case-insensitive), front-most first."""
    needle = query.casefold()
    return [window for window in windows if needle in window.title.casefold()]


def grab(bbox: Bbox) -> Image.Image:
    """Pixels of the given screen rectangle, as RGB."""
    return ImageGrab.grab(bbox=bbox, all_screens=True).convert("RGB")


def print_window(window: WindowInfo) -> Image.Image | None:
    """Render the window itself, even when other windows cover it. ``None`` if Windows refuses."""
    width, height = window.outer_size
    screen_dc = _user32.GetDC(None)
    memory_dc = _gdi32.CreateCompatibleDC(screen_dc)
    bitmap = _gdi32.CreateCompatibleBitmap(screen_dc, width, height)
    previous = _gdi32.SelectObject(memory_dc, bitmap)
    try:
        if not _user32.PrintWindow(window.handle, memory_dc, PW_RENDERFULLCONTENT):
            return None
        # Negative height: rows are stored top-down, which is what Pillow expects below.
        header = _BitmapInfoHeader(
            biSize=ctypes.sizeof(_BitmapInfoHeader),
            biWidth=width,
            biHeight=-height,
            biPlanes=1,
            biBitCount=32,
            biCompression=BI_RGB,
        )
        pixels = ctypes.create_string_buffer(width * height * 4)
        _gdi32.GetDIBits(memory_dc, bitmap, 0, height, pixels, ctypes.byref(header), DIB_RGB_COLORS)
        full = Image.frombuffer("RGB", (width, height), pixels, "raw", "BGRX", 0, 1)
    finally:
        _gdi32.SelectObject(memory_dc, previous)
        _gdi32.DeleteObject(bitmap)
        _gdi32.DeleteDC(memory_dc)
        _user32.ReleaseDC(None, screen_dc)

    # The visible frame sits inside the full rectangle, minus the invisible border.
    left = window.bbox[0] - window.outer[0]
    top = window.bbox[1] - window.outer[1]
    return full.crop((left, top, left + window.size[0], top + window.size[1]))


def capture_window(window: WindowInfo) -> Image.Image:
    image = print_window(window)
    if image is None:
        print("Avertissement : PrintWindow a échoué, capture des pixels de l'écran (fenêtre visible requise).")
        return grab(window.bbox)
    return image


def unique_output_path(folder: Path, now: datetime) -> Path:
    """``AAAAMMJJ_HHMMSS.png``, with a numeric suffix if two shots land in the same second."""
    stem = now.strftime("%Y%m%d_%H%M%S")
    path = folder / f"{stem}.png"
    counter = 2
    while path.exists():
        path = folder / f"{stem}_{counter}.png"
        counter += 1
    return path


def print_listing() -> None:
    print("Écrans (utilisez --screen N) :")
    for screen in list_screens():
        width, height = screen.size
        if screen.index == 0:
            label = "tous les écrans"
        else:
            label = "principal" if screen.primary else "secondaire"
        print(f"  [{screen.index}] {width}x{height} à ({screen.bbox[0]}, {screen.bbox[1]}) - {label}")
    print("\nFenêtres visibles, premier plan en premier (utilisez --window TEXTE) :")
    for window in list_windows():
        width, height = window.size
        print(f"  {window.title}  [{width}x{height}]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--list", action="store_true", help="lister les écrans et les fenêtres visibles")
    target.add_argument("--window", metavar="TEXTE", help="capturer la fenêtre dont le titre contient TEXTE")
    target.add_argument("--screen", type=int, metavar="N", help="capturer l'écran N (0 = tous les écrans)")
    parser.add_argument("-o", "--output", type=Path, help="fichier PNG de sortie")
    parser.add_argument("--delay", type=float, default=0.0, metavar="SECONDES", help="attendre avant la capture")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if sys.platform != "win32":
        print("Ce script fonctionne uniquement sous Windows.", file=sys.stderr)
        return 1
    _enable_dpi_awareness()
    args = build_parser().parse_args(argv)
    if args.list:
        print_listing()
        return 0

    window: WindowInfo | None = None
    try:
        if args.window is not None:
            matches = find_windows(args.window, list_windows())
            if not matches:
                raise LookupError(f"Aucune fenêtre visible ne correspond à {args.window!r} (voir --list)")
            if len(matches) > 1:
                print(f"{len(matches)} fenêtres correspondent, la première est utilisée : {matches[0].title}")
            window = matches[0]
        else:
            screens = list_screens()
            if not 0 <= args.screen < len(screens):
                raise LookupError(f"Écran {args.screen} inexistant (écrans 0 à {len(screens) - 1})")
            screen = screens[args.screen]

        if args.delay > 0:
            print(f"Capture dans {args.delay:g} s...")
            time.sleep(args.delay)

        image = capture_window(window) if window is not None else grab(screen.bbox)
        output = (args.output or unique_output_path(DEFAULT_OUTPUT_DIR, datetime.now())).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
    except (LookupError, OSError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1

    print(f"Capture enregistrée : {output} ({image.width}x{image.height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
