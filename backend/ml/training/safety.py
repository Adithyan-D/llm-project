"""Memory and OOM guards for CPU training.

CPU torch raises a ``RuntimeError`` from the allocator instead of a dedicated
exception type, and a failed allocation can leave the process in a bad state.
The training loop therefore checks an analytic budget *before* the first step
and converts allocator failures into one actionable error message.
"""

from __future__ import annotations

import ctypes
import os
import sys

OOM_MARKERS = (
    "out of memory",
    "can't allocate memory",
    "cannot allocate memory",
    "not enough memory",
    "defaultcpuallocator",
)


class InsufficientMemoryError(RuntimeError):
    """Raised before training when the estimated footprint exceeds the budget."""


class TrainingOOMError(RuntimeError):
    """Raised when torch fails to allocate during training."""


def is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in OOM_MARKERS)


def available_memory_bytes() -> int | None:
    """Best-effort available physical memory, or ``None`` if unknown.

    Windows uses ``GlobalMemoryStatusEx``; POSIX uses ``sysconf``. Some
    containers expose neither, and that is fine: the caller skips the check.
    """
    try:
        if sys.platform == "win32":
            return _windows_available_memory()
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages) * int(page_size)
    except (OSError, AttributeError, ValueError):
        return None


def _windows_available_memory() -> int | None:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullAvailPhys)


def ensure_memory_budget(
    estimated_bytes: int,
    available_bytes: int,
    *,
    max_fraction: float = 0.7,
    context: str = "",
) -> None:
    """Refuse to start a run that would use more than ``max_fraction`` of RAM.

    The estimate is deliberately conservative, but leaving a margin matters:
    allocator slack, the validation batches and the OS all share the same RAM.
    """
    if not 0.0 < max_fraction <= 1.0:
        raise ValueError("max_fraction must be in (0, 1]")
    if estimated_bytes < 0 or available_bytes < 0:
        raise ValueError("byte counts must be >= 0")
    limit = int(available_bytes * max_fraction)
    if estimated_bytes > limit:
        suffix = f" ({context})" if context else ""
        raise InsufficientMemoryError(
            f"estimated training footprint {_human(estimated_bytes)} exceeds the "
            f"{max_fraction:.0%} budget of {_human(limit)} available RAM{suffix}. "
            "Reduce batch_size, context_length or grad_accum_steps, or pass "
            "--force to override the guard."
        )


def _human(num_bytes: int | float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GiB"


__all__ = [
    "InsufficientMemoryError",
    "TrainingOOMError",
    "available_memory_bytes",
    "ensure_memory_budget",
    "is_oom_error",
]
