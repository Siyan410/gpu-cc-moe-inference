from __future__ import annotations

from pathlib import Path


DEFAULT_PROMPTS = [
    "Explain why GPU confidential computing matters for model inference.",
    "Write a short technical summary of mixture-of-experts routing.",
    "List three practical risks in deploying large language models.",
    "Compare bandwidth-bound and compute-bound neural network operators.",
]


def load_prompts(path: str | None) -> list[str]:
    if path is None:
        return list(DEFAULT_PROMPTS)
    prompts = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts
