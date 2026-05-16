from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import write_json


def _run_text(args: list[str], timeout: int = 10) -> str | None:
    if shutil.which(args[0]) is None:
        return None
    try:
        proc = subprocess.run(
            args,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip()


def collect_env() -> dict[str, Any]:
    env: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "nvidia_smi": _run_text(["nvidia-smi"]),
        "nvidia_smi_query": _run_text(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,pci.bus_id,pcie.link.gen.gpucurrent,pcie.link.width.current,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "nvidia_smi_pci": _run_text(["nvidia-smi", "-q", "-d", "PCI"]),
    }
    try:
        import torch  # type: ignore

        env["torch"] = {
            "version": getattr(torch, "__version__", None),
            "cuda_runtime": getattr(getattr(torch, "version", None), "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        }
    except Exception as exc:  # pragma: no cover - depends on host packages.
        env["torch"] = {"available": False, "error": str(exc)}
    try:
        import transformers  # type: ignore

        env["transformers"] = {"version": getattr(transformers, "__version__", None)}
    except Exception:
        env["transformers"] = {"available": False}
    return env


def write_env(out_dir: str | Path) -> None:
    write_json(Path(out_dir) / "env.json", collect_env())


def print_env_json() -> None:
    print(json.dumps(collect_env(), indent=2, sort_keys=True))
