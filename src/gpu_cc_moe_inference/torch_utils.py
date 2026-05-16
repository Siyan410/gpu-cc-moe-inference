from __future__ import annotations

from typing import Any


def import_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - package dependent.
        raise RuntimeError("PyTorch is required for this command") from exc
    return torch


def require_cuda() -> Any:
    torch = import_torch()
    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:
        raise RuntimeError(f"PyTorch CUDA initialization failed: {exc}") from exc
    if not available:
        cuda_runtime = getattr(getattr(torch, "version", None), "cuda", None)
        version = getattr(torch, "__version__", "unknown")
        raise RuntimeError(
            "PyTorch cannot use CUDA in this environment. "
            f"torch={version}, torch.version.cuda={cuda_runtime}. "
            "Check that the PyTorch CUDA wheel is compatible with the installed NVIDIA driver."
        )
    return torch
