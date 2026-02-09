from __future__ import annotations

import os
import platform
import shutil
import subprocess
from typing import Any


def _has_command(name: str) -> bool:
    return shutil.which(name) is not None


def has_nvidia_gpu() -> bool:
    if _has_command("nvidia-smi"):
        try:
            completed = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return completed.returncode == 0 and "GPU" in completed.stdout
        except Exception:
            return False

    # Fallback heuristics for headless environments.
    if os.path.exists("/proc/driver/nvidia/version"):
        return True
    return False


def resolve_xgb_device(mode: str = "auto") -> tuple[str, str]:
    requested = str(mode or "auto").lower()

    if requested == "cpu":
        return "cpu", "forced cpu mode"

    if requested == "cuda":
        if has_nvidia_gpu():
            return "cuda", "cuda requested and NVIDIA GPU detected"
        return "cpu", "cuda requested but no NVIDIA GPU detected; fallback to cpu"

    # auto mode
    if has_nvidia_gpu():
        return "cuda", "auto detected NVIDIA GPU"
    return "cpu", "auto mode without NVIDIA GPU"


def hardware_summary() -> dict[str, Any]:
    system = platform.system()
    machine = platform.machine()
    nvidia = has_nvidia_gpu()

    return {
        "system": system,
        "machine": machine,
        "processor": platform.processor(),
        "has_nvidia_gpu": nvidia,
        "is_apple_silicon": system == "Darwin" and machine in {"arm64", "aarch64"},
    }
