from __future__ import annotations

import ctypes
import os
import subprocess
from typing import Callable, Literal


ResourceMode = Literal["balanced", "max"]

BELOW_NORMAL_PRIORITY_CLASS = 0x00004000


def resource_mode_is_balanced(mode: ResourceMode | str | None) -> bool:
    return mode != "max"


def process_creationflags(low_priority: bool) -> int:
    if not low_priority or os.name != "nt":
        return 0
    return int(getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", BELOW_NORMAL_PRIORITY_CLASS))


def lower_current_process_priority(low_priority: bool) -> Callable[[], None]:
    if not low_priority or os.name != "nt":
        return lambda: None

    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        current_process = kernel32.GetCurrentProcess()
        previous_priority = int(kernel32.GetPriorityClass(current_process))
        if previous_priority:
            kernel32.SetPriorityClass(current_process, BELOW_NORMAL_PRIORITY_CLASS)

        def restore() -> None:
            if previous_priority:
                kernel32.SetPriorityClass(current_process, previous_priority)

        return restore
    except Exception:
        return lambda: None
