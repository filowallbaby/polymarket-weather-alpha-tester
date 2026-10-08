from __future__ import annotations

import ctypes
import os
from typing import Any


_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def inhibit_system_sleep() -> dict[str, Any]:

    if os.name != "nt":
        return {"supported": False, "active": False, "method": None}
    result = ctypes.windll.kernel32.SetThreadExecutionState(
        _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED
    )
    return {
        "supported": True,
        "active": bool(result),
        "method": "SetThreadExecutionState(ES_CONTINUOUS|ES_SYSTEM_REQUIRED)",
    }


def release_system_sleep_inhibition() -> bool:
    if os.name != "nt":
        return True
    result = ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
    return bool(result)
