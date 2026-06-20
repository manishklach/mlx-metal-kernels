from __future__ import annotations

from datetime import datetime, timezone
import os
import platform
import subprocess
import sys


def _try_command(args):
    try:
        out = subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True)
        return out.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _mlx_version():
    try:
        import mlx  # type: ignore

        return getattr(mlx, "__version__", None)
    except Exception:  # noqa: BLE001
        return None


def collect_system_info():
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "python_version": sys.version.replace("\n", " "),
        "macos_version": platform.mac_ver()[0] or None,
        "mlx_version": _mlx_version(),
        "chip_info": _try_command(["sysctl", "-n", "machdep.cpu.brand_string"]) or _try_command(["system_profiler", "SPHardwareDataType"]),
        "env": {
            "MLX_METAL_USE_SPECIALIZED": os.environ.get("MLX_METAL_USE_SPECIALIZED"),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        },
    }
