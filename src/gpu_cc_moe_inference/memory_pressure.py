from __future__ import annotations

import argparse
import gc
import time
import traceback
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from .env import write_env
from .io import append_jsonl, ensure_dir
from .prompting import load_prompts
from .torch_utils import require_cuda


SCHEMA_VERSION = "memory-pressure-v1"


def parse_float_list(value: str) -> list[float]:
    items = [float(item) for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one numeric value")
    if any(item < 0 for item in items):
        raise argparse.ArgumentTypeError("reservation values must be non-negative")
    return items


def gpu_memory_snapshot(torch: Any, note: str) -> dict[str, Any]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "note": note,
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "used_bytes": int(total_bytes - free_bytes),
        "torch_allocated_bytes": int(torch.cuda.memory_allocated()),
        "torch_reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_torch_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_torch_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def reserve_gpu_memory(
    torch: Any,
    reserve_gb: float,
    *,
    chunk_mb: int,
    touch: bool,
) -> list[Any]:
    tensors: list[Any] = []
    remaining = int(reserve_gb * 1024**3)
    chunk_bytes = max(1, int(chunk_mb * 1024**2))
    while remaining > 0:
        size = min(remaining, chunk_bytes)
        tensor = torch.empty(size, dtype=torch.uint8, device="cuda")
        if touch:
            tensor[0] = 0
        tensors.append(tensor)
        remaining -= size
    torch.cuda.synchronize()
    return tensors


def summarize_parameter_devices(model: Any, *, include_parameter_bytes: bool) -> dict[str, Any]:
    device_map = getattr(model, "hf_device_map", None)
    device_map_counts: Counter[str] = Counter()
    if isinstance(device_map, dict):
        device_map_counts.update(str(device) for device in device_map.values())
    if not include_parameter_bytes:
        return {
            "hf_device_map": device_map,
            "hf_device_map_device_counts": dict(device_map_counts),
        }

    counts: Counter[str] = Counter()
    bytes_by_device: Counter[str] = Counter()
    module_devices: dict[str, str] = {}
    for name, parameter in model.named_parameters():
        device = str(parameter.device)
        counts[device] += parameter.numel()
        bytes_by_device[device] += parameter.numel() * parameter.element_size()
        module = name.rsplit(".", 1)[0]
        module_devices.setdefault(module, device)
    return {
        "parameter_count_by_device": dict(counts),
        "parameter_bytes_by_device": dict(bytes_by_device),
        "hf_device_map": device_map,
        "hf_device_map_device_counts": dict(device_map_counts),
        "module_device_sample": dict(list(module_devices.items())[:80]),
    }


def has_cpu_or_disk_offload(row: dict[str, Any]) -> bool:
    devices = set(row.get("parameter_bytes_by_device", {}))
    if any(device == "cpu" or device.startswith("disk") for device in devices):
        return True
    device_map = row.get("hf_device_map") or {}
    if isinstance(device_map, dict):
        return any(str(device) in {"cpu", "disk"} for device in device_map.values())
    device_counts = row.get("hf_device_map_device_counts") or {}
    if isinstance(device_counts, dict):
        return any(str(device) in {"cpu", "disk"} for device in device_counts)
    return False


def first_prompt(prompts_path: str | None) -> str:
    prompts = load_prompts(prompts_path)
    return prompts[0]


def run_one(args: argparse.Namespace, reserve_gb: float) -> dict[str, Any]:
    torch = require_cuda()
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    snapshots = [gpu_memory_snapshot(torch, "before_reservation")]
    reservation: list[Any] = []
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "reserve_gb": reserve_gb,
        "model": args.model,
        "device_map_arg": args.device_map,
        "dtype_arg": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "status": "unknown",
    }
    model = None
    try:
        reservation = reserve_gpu_memory(
            torch,
            reserve_gb,
            chunk_mb=args.reserve_chunk_mb,
            touch=args.touch_reservation,
        )
        snapshots.append(gpu_memory_snapshot(torch, "after_reservation"))
        tokenizer_start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=args.trust_remote_code
        )
        row["tokenizer_load_ms"] = (time.perf_counter() - tokenizer_start) * 1000
        prompt = first_prompt(args.prompts)
        encoded_cpu = tokenizer(prompt, return_tensors="pt")
        row["prompt_chars"] = len(prompt)
        row["input_tokens"] = int(encoded_cpu["input_ids"].shape[-1])

        load_start = time.perf_counter()
        kwargs = {
            "torch_dtype": torch.bfloat16 if args.dtype == "bf16" else "auto",
            "device_map": args.device_map,
            "trust_remote_code": args.trust_remote_code,
        }
        if args.offload_folder:
            offload_folder = Path(args.offload_folder)
            offload_folder.mkdir(parents=True, exist_ok=True)
            kwargs["offload_folder"] = str(offload_folder)
        model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
        model.eval()
        row["model_load_ms"] = (time.perf_counter() - load_start) * 1000
        snapshots.append(gpu_memory_snapshot(torch, "after_model_load"))
        row.update(
            summarize_parameter_devices(
                model, include_parameter_bytes=args.include_parameter_bytes
            )
        )
        row["cpu_or_disk_offload_detected"] = has_cpu_or_disk_offload(row)

        if row["cpu_or_disk_offload_detected"] and args.skip_generate_when_offloaded:
            row["status"] = "loaded_with_offload_generate_skipped"
            row["skip_reason"] = "cpu_or_disk_offload_detected"
            return row

        if not args.run_generate:
            row["status"] = "loaded_generate_not_requested"
            return row

        target_device = getattr(model, "device", torch.device("cuda"))
        h2d_start = time.perf_counter()
        encoded = {key: value.to(target_device) for key, value in encoded_cpu.items()}
        torch.cuda.synchronize()
        row["input_h2d_ms"] = (time.perf_counter() - h2d_start) * 1000
        generate_start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **encoded, max_new_tokens=args.max_new_tokens, do_sample=False
            )
        torch.cuda.synchronize()
        row["generate_ms"] = (time.perf_counter() - generate_start) * 1000
        row["output_tokens"] = int(output.shape[-1])
        snapshots.append(gpu_memory_snapshot(torch, "after_generate"))
        row["status"] = "ok"
    except Exception as exc:  # pragma: no cover - intentionally observes failures.
        row["status"] = "error"
        row["error_type"] = type(exc).__name__
        row["error_message"] = str(exc)
        row["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
        try:
            snapshots.append(gpu_memory_snapshot(torch, "after_error"))
        except Exception:
            pass
    finally:
        del model
        reservation.clear()
        gc.collect()
        torch.cuda.empty_cache()
        snapshots.append(gpu_memory_snapshot(torch, "after_cleanup"))
        row["memory_snapshots"] = snapshots
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Observe current inference behavior under artificial GPU memory pressure."
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts")
    parser.add_argument("--run-id", default=str(uuid.uuid4()))
    parser.add_argument(
        "--reserve-gb",
        type=parse_float_list,
        default=parse_float_list("0,20,35,45"),
        help="Comma-separated GPU memory reservation sizes in GiB.",
    )
    parser.add_argument("--reserve-chunk-mb", type=int, default=512)
    parser.add_argument("--touch-reservation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument(
        "--run-generate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run one short generate after loading. Disabled by default to avoid slow CPU-offload runs.",
    )
    parser.add_argument(
        "--skip-generate-when-offloaded",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip generate if parameters were dispatched to CPU or disk.",
    )
    parser.add_argument("--dtype", choices=["bf16", "auto"], default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--offload-folder")
    parser.add_argument(
        "--include-parameter-bytes",
        action="store_true",
        help="Slow path: iterate all parameters to count bytes by device.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = ensure_dir(args.out)
    write_env(out_dir)
    output_path = out_dir / "memory_pressure_observations.jsonl"
    if output_path.exists():
        output_path.unlink()
    for reserve_gb in args.reserve_gb:
        append_jsonl(output_path, run_one(args, reserve_gb))
    return 0
