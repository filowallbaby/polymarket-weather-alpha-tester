from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from pathlib import Path
from typing import Any


def process_memory() -> dict[str, int | None]:


    if os.name == "nt":
        return _windows_process_memory()
    return _proc_process_memory()


def _windows_process_memory() -> dict[str, int | None]:
    size_t = ctypes.c_size_t

    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", size_t),
            ("WorkingSetSize", size_t),
            ("QuotaPeakPagedPoolUsage", size_t),
            ("QuotaPagedPoolUsage", size_t),
            ("QuotaPeakNonPagedPoolUsage", size_t),
            ("QuotaNonPagedPoolUsage", size_t),
            ("PagefileUsage", size_t),
            ("PeakPagefileUsage", size_t),
            ("PrivateUsage", size_t),
        ]

    counters = ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCountersEx),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        return _empty_memory()
    return {
        "pid": os.getpid(),
        "private_bytes": int(counters.PrivateUsage),
        "working_set_bytes": int(counters.WorkingSetSize),
        "peak_private_bytes": int(counters.PeakPagefileUsage),
        "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
        "virtual_bytes": None,
    }


def _proc_process_memory() -> dict[str, int | None]:
    values: dict[str, int] = {}
    status = Path("/proc/self/status")
    if status.exists():
        try:
            for line in status.read_text(encoding="utf-8").splitlines():
                name, separator, raw = line.partition(":")
                if not separator:
                    continue
                parts = raw.strip().split()
                if parts and parts[0].isdigit():
                    multiplier = 1024 if len(parts) > 1 and parts[1].lower() == "kb" else 1
                    values[name] = int(parts[0]) * multiplier
        except OSError:
            pass
    return {
        "pid": os.getpid(),
        "private_bytes": values.get("VmData"),
        "working_set_bytes": values.get("VmRSS"),
        "peak_private_bytes": values.get("VmPeak"),
        "peak_working_set_bytes": values.get("VmHWM"),
        "virtual_bytes": values.get("VmSize"),
    }


def _empty_memory() -> dict[str, Any]:
    return {
        "pid": os.getpid(),
        "private_bytes": None,
        "working_set_bytes": None,
        "peak_private_bytes": None,
        "peak_working_set_bytes": None,
        "virtual_bytes": None,
    }
