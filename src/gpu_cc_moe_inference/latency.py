from __future__ import annotations

import argparse
import re
import time
import uuid
from collections import defaultdict
from typing import Any

from .env import write_env
from .io import ensure_dir, write_jsonl
from .prompting import load_prompts
from .torch_utils import require_cuda


SCHEMA_VERSION = "moe-latency-v1"


def categorize_module(name: str) -> str | None:
    lowered = name.lower()
    if "attn" in lowered or "attention" in lowered:
        return "attention"
    if "router" in lowered or re.search(r"(^|\.)gate($|\.)", lowered):
        return "router"
    if "moe" in lowered or "expert" in lowered or "sparsemixer" in lowered:
        return "moe_ffn"
    return None


class ModuleTimer:
    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self.stack: dict[str, list[tuple[float, Any, Any]]] = defaultdict(list)
        self.rows: list[dict[str, Any]] = []

    def pre(self, name: str, category: str):
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            start_event = self.torch.cuda.Event(enable_timing=True)
            end_event = self.torch.cuda.Event(enable_timing=True)
            start_event.record()
            shape = None
            if inputs and hasattr(inputs[0], "shape"):
                shape = list(inputs[0].shape)
            self.stack[name].append((time.perf_counter(), start_event, (end_event, shape, category)))

        return hook

    def post(self, name: str):
        def hook(_module: Any, _inputs: tuple[Any, ...], _output: Any) -> None:
            if not self.stack[name]:
                return
            start_wall, start_event, payload = self.stack[name].pop()
            end_event, shape, category = payload
            end_event.record()
            self.rows.append(
                {
                    "module": name,
                    "category": category,
                    "input_shape": shape,
                    "cpu_wall_ms": (time.perf_counter() - start_wall) * 1000,
                    "cuda_event_start": start_event,
                    "cuda_event_end": end_event,
                }
            )

        return hook

    def finalize(self) -> list[dict[str, Any]]:
        self.torch.cuda.synchronize()
        finalized = []
        for row in self.rows:
            start = row.pop("cuda_event_start")
            end = row.pop("cuda_event_end")
            row["cuda_event_ms"] = float(start.elapsed_time(end))
            finalized.append(row)
        return finalized


def latency_run(args: argparse.Namespace) -> int:
    torch = require_cuda()
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    out_dir = ensure_dir(args.out)
    write_env(out_dir)
    prompts = load_prompts(args.prompts)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.dtype == "bf16" else "auto",
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    timer = ModuleTimer(torch)
    hooks = []
    for name, module in model.named_modules():
        category = categorize_module(name)
        if category is None:
            continue
        hooks.append(module.register_forward_pre_hook(timer.pre(name, category)))
        hooks.append(module.register_forward_hook(timer.post(name)))
    rows: list[dict[str, Any]] = []
    try:
        with torch.inference_mode():
            for prompt_id, prompt in enumerate(prompts):
                token_start = time.perf_counter()
                encoded = tokenizer(prompt, return_tensors="pt")
                tokenization_ms = (time.perf_counter() - token_start) * 1000
                input_bytes = sum(value.numel() * value.element_size() for value in encoded.values())
                h2d_start = time.perf_counter()
                encoded = {key: value.to(model.device) for key, value in encoded.items()}
                torch.cuda.synchronize()
                h2d_ms = (time.perf_counter() - h2d_start) * 1000
                gen_start = time.perf_counter()
                output = model.generate(**encoded, max_new_tokens=args.max_new_tokens, do_sample=False)
                torch.cuda.synchronize()
                generate_ms = (time.perf_counter() - gen_start) * 1000
                decode_start = time.perf_counter()
                tokenizer.batch_decode(output, skip_special_tokens=True)
                decode_output_ms = (time.perf_counter() - decode_start) * 1000
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "run_id": args.run_id,
                        "prompt_id": str(prompt_id),
                        "category": "end_to_end",
                        "tokenization_ms": tokenization_ms,
                        "input_h2d_bytes": input_bytes,
                        "input_h2d_ms": h2d_ms,
                        "generate_cpu_wall_ms": generate_ms,
                        "decode_output_ms": decode_output_ms,
                        "output_tokens": int(output.shape[-1]),
                        "security_use": "performance_only_not_sidechannel_conclusion_feature",
                    }
                )
    finally:
        for hook in hooks:
            hook.remove()
    for row in timer.finalize():
        row.update(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": args.run_id,
                "security_use": "performance_only_not_sidechannel_conclusion_feature",
            }
        )
        rows.append(row)
    write_jsonl(out_dir / "moe_latency_measurements.jsonl", rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Instrument MoE inference latency.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts")
    parser.add_argument("--run-id", default=str(uuid.uuid4()))
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dtype", choices=["bf16", "auto"], default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return latency_run(args)
