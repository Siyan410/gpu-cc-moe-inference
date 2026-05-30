from __future__ import annotations

import argparse
import csv
import math
import statistics
import threading
import time
import uuid
from dataclasses import dataclass
from queue import Empty
from pathlib import Path
from typing import Any, Iterable

from .env import write_env
from .io import ensure_dir, read_jsonl, write_json, write_jsonl
from .torch_utils import require_cuda


SCHEMA_VERSION = "overhead-amplification-v1"
SUMMARY_VERSION = "overhead-amplification-summary-v1"
COMPARISON_VERSION = "overhead-amplification-comparison-v1"


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    family: str
    prompt: str
    max_new_tokens: int
    target_input_tokens: int | None = None
    streaming: bool = False


def parse_int_list(value: str) -> list[int]:
    items = [int(item) for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(item <= 0 for item in items):
        raise argparse.ArgumentTypeError("all values must be positive")
    return items


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def metric_summary(values: list[float]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "p50": percentile(clean, 0.50),
        "p95": percentile(clean, 0.95),
        "p99": percentile(clean, 0.99),
    }


def _prompt_baseline() -> str:
    return (
        "Explain GPU confidential computing for large language model inference "
        "in a concise technical answer. Include the main performance tradeoffs."
    )


def _prompt_decode_heavy() -> str:
    return (
        "Write a long numbered technical report about confidential GPU LLM serving. "
        "Each item must be different, concrete, and one sentence long. Continue "
        "the numbered list without summarizing."
    )


def _prompt_structured_json() -> str:
    return (
        "Return strictly valid JSON only. The JSON value must be an object with a "
        "field named items. items must be an array of objects. Each object must "
        "contain id, name, description, tags, dependencies, and risk fields. Do "
        "not output markdown or any text outside JSON."
    )


def _prompt_streaming_steps() -> str:
    return (
        "Produce a long step-by-step operational checklist for benchmarking an "
        "LLM serving stack. Put each step on its own line and keep going until "
        "the requested output budget is exhausted."
    )


def _long_context_prompt(target_input_tokens: int) -> str:
    paragraph = (
        "Confidential GPU serving keeps model weights, activations, prompts, and "
        "intermediate tensors inside protected execution environments. The service "
        "still needs scheduling, batching, KV cache management, token streaming, "
        "and host-device coordination. "
    )
    return (
        f"You are given a target-length technical context of about {target_input_tokens} "
        "tokens. Use only the context when "
        "answering the final question.\n\n"
        f"{paragraph}\n\n"
        "Question: Identify the availability risks that arise when KV cache "
        "locality is disrupted under confidential GPU serving."
    )


def builtin_workloads(
    *,
    suite: str,
    selected: set[str] | None = None,
    decode_output_tokens: list[int] | None = None,
    structured_output_tokens: list[int] | None = None,
    streaming_output_tokens: list[int] | None = None,
    long_context_tokens: list[int] | None = None,
) -> list[WorkloadSpec]:
    if suite == "smoke":
        decode_output_tokens = decode_output_tokens or [64]
        structured_output_tokens = structured_output_tokens or [64]
        streaming_output_tokens = streaming_output_tokens or [64]
        long_context_tokens = long_context_tokens or [512]
        baseline_tokens = 32
    elif suite == "full":
        decode_output_tokens = decode_output_tokens or [512, 1024, 2048]
        structured_output_tokens = structured_output_tokens or [512, 1024]
        streaming_output_tokens = streaming_output_tokens or [512, 1024]
        long_context_tokens = long_context_tokens or [4096, 16384, 32768]
        baseline_tokens = 128
    else:
        decode_output_tokens = decode_output_tokens or [512, 1024]
        structured_output_tokens = structured_output_tokens or [512]
        streaming_output_tokens = streaming_output_tokens or [512]
        long_context_tokens = long_context_tokens or [4096]
        baseline_tokens = 128

    specs: list[WorkloadSpec] = [
        WorkloadSpec(
            name=f"baseline_chat_out{baseline_tokens}",
            family="baseline_chat",
            prompt=_prompt_baseline(),
            max_new_tokens=baseline_tokens,
        )
    ]
    for tokens in decode_output_tokens:
        specs.append(
            WorkloadSpec(
                name=f"decode_heavy_out{tokens}",
                family="decode_heavy",
                prompt=_prompt_decode_heavy(),
                max_new_tokens=tokens,
            )
        )
    for tokens in structured_output_tokens:
        specs.append(
            WorkloadSpec(
                name=f"structured_json_prompt_out{tokens}",
                family="structured_json_prompt",
                prompt=_prompt_structured_json(),
                max_new_tokens=tokens,
            )
        )
    for tokens in long_context_tokens:
        specs.append(
            WorkloadSpec(
                name=f"long_context_{tokens}_out128",
                family="long_context",
                prompt=_long_context_prompt(tokens),
                target_input_tokens=tokens,
                max_new_tokens=128,
            )
        )
    for tokens in streaming_output_tokens:
        specs.append(
            WorkloadSpec(
                name=f"streaming_steps_out{tokens}",
                family="streaming_steps",
                prompt=_prompt_streaming_steps(),
                max_new_tokens=tokens,
                streaming=True,
            )
        )

    if selected is None:
        return specs
    return [spec for spec in specs if spec.family in selected or spec.name in selected]


def fit_prompt_to_token_budget(tokenizer: Any, spec: WorkloadSpec) -> str:
    if spec.target_input_tokens is None:
        return spec.prompt
    suffix = (
        "\n\nFinal instruction: answer the question using citations to at least "
        "three different parts of the context."
    )
    prompt = spec.prompt + suffix
    length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if length >= spec.target_input_tokens:
        return prompt
    filler = (
        " Confidential serving systems must preserve locality across scheduler "
        "queues, KV pages, tensor placements, and network responses."
    )
    while length < spec.target_input_tokens:
        prompt += filler
        length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    return prompt


def _cycle_batch(prompt: str, batch_size: int, repeat_index: int) -> list[str]:
    return [
        f"{prompt}\n\nRun variant: {repeat_index}-{index}."
        for index in range(batch_size)
    ]


def _tensor_bytes(tensors: dict[str, Any]) -> int:
    return int(sum(value.numel() * value.element_size() for value in tensors.values()))


def _to_device(tensors: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in tensors.items()}


def _generation_kwargs(args: argparse.Namespace, tokenizer: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"do_sample": args.do_sample}
    if getattr(tokenizer, "pad_token_id", None) is not None:
        kwargs["pad_token_id"] = tokenizer.pad_token_id
    if getattr(tokenizer, "eos_token_id", None) is not None:
        kwargs["eos_token_id"] = tokenizer.eos_token_id
    return kwargs


def _input_lengths(encoded: dict[str, Any]) -> list[int]:
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        return [int(item) for item in attention_mask.sum(dim=1).tolist()]
    return [int(encoded["input_ids"].shape[-1])] * int(encoded["input_ids"].shape[0])


def _load_transformers_model(args: argparse.Namespace) -> tuple[Any, Any, float]:
    torch = require_cuda()
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.dtype == "bf16" else "auto",
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    return tokenizer, model, (time.perf_counter() - load_start) * 1000


def _warmup_model(args: argparse.Namespace, tokenizer: Any, model: Any) -> None:
    if args.warmup <= 0:
        return
    torch = require_cuda()
    target_device = getattr(model, "device", torch.device("cuda"))
    kwargs = _generation_kwargs(args, tokenizer)
    prompt = "Warm up the model with a short deterministic response."
    with torch.inference_mode():
        for _ in range(args.warmup):
            encoded = tokenizer([prompt], return_tensors="pt", padding=True)
            encoded = _to_device(encoded, target_device)
            model.generate(**encoded, max_new_tokens=1, **kwargs)
            torch.cuda.synchronize()


def measure_nonstreaming(
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    model: Any,
    spec: WorkloadSpec,
    prompt: str,
    batch_size: int,
    repeat_index: int,
) -> dict[str, Any]:
    torch = require_cuda()
    target_device = getattr(model, "device", torch.device("cuda"))
    gen_kwargs = _generation_kwargs(args, tokenizer)
    prompts = _cycle_batch(prompt, batch_size, repeat_index)

    token_start = time.perf_counter()
    encoded_cpu = tokenizer(prompts, return_tensors="pt", padding=True)
    tokenization_ms = (time.perf_counter() - token_start) * 1000
    input_lengths = _input_lengths(encoded_cpu)
    input_bytes = _tensor_bytes(encoded_cpu)

    h2d_start = time.perf_counter()
    encoded = _to_device(encoded_cpu, target_device)
    torch.cuda.synchronize()
    input_h2d_ms = (time.perf_counter() - h2d_start) * 1000

    ttft_proxy_ms = None
    if args.measure_ttft_proxy:
        ttft_start = time.perf_counter()
        with torch.inference_mode():
            model.generate(**encoded, max_new_tokens=1, **gen_kwargs)
        torch.cuda.synchronize()
        ttft_proxy_ms = (time.perf_counter() - ttft_start) * 1000

    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **encoded, max_new_tokens=spec.max_new_tokens, **gen_kwargs
        )
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000

    padded_input_tokens = int(encoded["input_ids"].shape[-1])
    output_tokens_per_request = [int(output.shape[-1])] * batch_size
    new_tokens_per_request = [
        max(0, int(output.shape[-1]) - padded_input_tokens)
        for _ in range(batch_size)
    ]
    generated_tokens_total = int(sum(new_tokens_per_request))
    tpot_ms = (
        latency_ms / generated_tokens_total if generated_tokens_total > 0 else None
    )
    tpot_excluding_ttft_proxy_ms = None
    if (
        ttft_proxy_ms is not None
        and generated_tokens_total > batch_size
        and latency_ms > ttft_proxy_ms
    ):
        tpot_excluding_ttft_proxy_ms = (
            latency_ms - ttft_proxy_ms
        ) / max(1, generated_tokens_total - batch_size)

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "cc_mode": args.cc_mode,
        "backend": "transformers",
        "workload": spec.name,
        "workload_family": spec.family,
        "repeat_index": repeat_index,
        "batch_size": batch_size,
        "streaming": False,
        "target_input_tokens": spec.target_input_tokens,
        "max_new_tokens": spec.max_new_tokens,
        "status": "ok",
        "input_tokens_per_request": input_lengths,
        "input_tokens_total": int(sum(input_lengths)),
        "input_h2d_bytes": input_bytes,
        "input_h2d_ms": input_h2d_ms,
        "tokenization_ms": tokenization_ms,
        "latency_ms": latency_ms,
        "ttft_ms": None,
        "ttft_proxy_ms": ttft_proxy_ms,
        "tpot_ms": tpot_ms,
        "tpot_excluding_ttft_proxy_ms": tpot_excluding_ttft_proxy_ms,
        "generated_tokens_per_request": new_tokens_per_request,
        "generated_tokens_total": generated_tokens_total,
        "output_tokens_per_request": output_tokens_per_request,
        "tokens_per_second": (
            generated_tokens_total / (latency_ms / 1000)
            if latency_ms > 0
            else None
        ),
    }


def measure_streaming(
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    model: Any,
    spec: WorkloadSpec,
    prompt: str,
    batch_size: int,
    repeat_index: int,
) -> dict[str, Any]:
    torch = require_cuda()
    if batch_size != 1:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": args.run_id,
            "cc_mode": args.cc_mode,
            "backend": "transformers",
            "workload": spec.name,
            "workload_family": spec.family,
            "repeat_index": repeat_index,
            "batch_size": batch_size,
            "streaming": True,
            "target_input_tokens": spec.target_input_tokens,
            "max_new_tokens": spec.max_new_tokens,
            "status": "skipped",
            "skip_reason": "transformers TextIteratorStreamer is measured only for batch_size=1",
        }
    from transformers import TextIteratorStreamer  # type: ignore

    target_device = getattr(model, "device", torch.device("cuda"))
    gen_kwargs = _generation_kwargs(args, tokenizer)
    prompts = _cycle_batch(prompt, batch_size, repeat_index)
    token_start = time.perf_counter()
    encoded_cpu = tokenizer(prompts, return_tensors="pt", padding=True)
    tokenization_ms = (time.perf_counter() - token_start) * 1000
    input_lengths = _input_lengths(encoded_cpu)
    input_bytes = _tensor_bytes(encoded_cpu)
    h2d_start = time.perf_counter()
    encoded = _to_device(encoded_cpu, target_device)
    torch.cuda.synchronize()
    input_h2d_ms = (time.perf_counter() - h2d_start) * 1000

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=0.5
    )
    generation_kwargs = {
        **encoded,
        "streamer": streamer,
        "max_new_tokens": spec.max_new_tokens,
        **gen_kwargs,
    }
    chunk_times_ms: list[float] = []
    chunk_chars: list[int] = []
    errors: list[str] = []

    def generate() -> None:
        try:
            with torch.inference_mode():
                model.generate(**generation_kwargs)
        except Exception as exc:  # pragma: no cover - depends on GPU runtime.
            errors.append(f"{type(exc).__name__}: {exc}")

    start = time.perf_counter()
    thread = threading.Thread(target=generate, daemon=True)
    thread.start()
    while True:
        try:
            chunk = next(streamer)
        except StopIteration:
            break
        except Empty:
            if not thread.is_alive():
                break
            continue
        now_ms = (time.perf_counter() - start) * 1000
        chunk_times_ms.append(now_ms)
        chunk_chars.append(len(chunk))
    thread.join()
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000
    if errors:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": args.run_id,
            "cc_mode": args.cc_mode,
            "backend": "transformers",
            "workload": spec.name,
            "workload_family": spec.family,
            "repeat_index": repeat_index,
            "batch_size": batch_size,
            "streaming": True,
            "target_input_tokens": spec.target_input_tokens,
            "max_new_tokens": spec.max_new_tokens,
            "status": "error",
            "error_message": errors[0],
        }

    padded_input_tokens = int(encoded["input_ids"].shape[-1])
    generated_tokens_total = spec.max_new_tokens
    ttft_ms = chunk_times_ms[0] if chunk_times_ms else None
    tpot_ms = None
    if generated_tokens_total > 0:
        tpot_ms = latency_ms / generated_tokens_total
    tpot_after_first_chunk_ms = None
    if ttft_ms is not None and generated_tokens_total > 1:
        tpot_after_first_chunk_ms = (
            latency_ms - ttft_ms
        ) / max(1, generated_tokens_total - 1)
    interarrival_ms = [
        chunk_times_ms[index] - chunk_times_ms[index - 1]
        for index in range(1, len(chunk_times_ms))
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "cc_mode": args.cc_mode,
        "backend": "transformers",
        "workload": spec.name,
        "workload_family": spec.family,
        "repeat_index": repeat_index,
        "batch_size": batch_size,
        "streaming": True,
        "target_input_tokens": spec.target_input_tokens,
        "max_new_tokens": spec.max_new_tokens,
        "status": "ok",
        "input_tokens_per_request": input_lengths,
        "input_tokens_total": int(sum(input_lengths)),
        "input_h2d_bytes": input_bytes,
        "input_h2d_ms": input_h2d_ms,
        "tokenization_ms": tokenization_ms,
        "latency_ms": latency_ms,
        "ttft_ms": ttft_ms,
        "ttft_proxy_ms": None,
        "tpot_ms": tpot_ms,
        "tpot_after_first_chunk_ms": tpot_after_first_chunk_ms,
        "generated_tokens_per_request": [generated_tokens_total],
        "generated_tokens_total": generated_tokens_total,
        "output_tokens_per_request": [padded_input_tokens + generated_tokens_total],
        "tokens_per_second": (
            generated_tokens_total / (latency_ms / 1000)
            if latency_ms > 0
            else None
        ),
        "stream_chunks": len(chunk_times_ms),
        "stream_chunk_chars_total": int(sum(chunk_chars)),
        "stream_interarrival_ms": metric_summary(interarrival_ms),
    }


def group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("backend"),
        row.get("workload"),
        row.get("workload_family"),
        row.get("batch_size"),
        row.get("streaming"),
        row.get("target_input_tokens"),
        row.get("max_new_tokens"),
    )


def group_key_dict(key: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "backend": key[0],
        "workload": key[1],
        "workload_family": key[2],
        "batch_size": key[3],
        "streaming": key[4],
        "target_input_tokens": key[5],
        "max_new_tokens": key[6],
    }


def summarize_measurements(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in ok_rows:
        grouped.setdefault(group_key(row), []).append(row)

    groups = []
    metric_names = [
        "latency_ms",
        "ttft_ms",
        "ttft_proxy_ms",
        "tpot_ms",
        "tpot_excluding_ttft_proxy_ms",
        "tpot_after_first_chunk_ms",
        "input_h2d_ms",
        "tokenization_ms",
        "tokens_per_second",
        "generated_tokens_total",
        "stream_chunks",
    ]
    for key, items in sorted(grouped.items(), key=lambda item: str(item[0])):
        metrics: dict[str, Any] = {}
        for metric in metric_names:
            values = [
                float(row[metric])
                for row in items
                if isinstance(row.get(metric), (int, float))
            ]
            metrics[metric] = metric_summary(values)
        group = group_key_dict(key)
        group.update({"rows": len(items), "metrics": metrics})
        groups.append(group)
    return {
        "schema_version": SUMMARY_VERSION,
        "rows": len(ok_rows),
        "groups": groups,
    }


def run_transformers(args: argparse.Namespace) -> int:
    out_dir = ensure_dir(args.out)
    write_env(out_dir)
    tokenizer, model, model_load_ms = _load_transformers_model(args)
    _warmup_model(args, tokenizer, model)

    selected = (
        {item.strip() for item in args.workloads.split(",") if item.strip()}
        if args.workloads
        else None
    )
    specs = builtin_workloads(
        suite=args.suite,
        selected=selected,
        decode_output_tokens=args.decode_output_tokens,
        structured_output_tokens=args.structured_output_tokens,
        streaming_output_tokens=args.streaming_output_tokens,
        long_context_tokens=args.long_context_tokens,
    )
    if not specs:
        raise ValueError(f"no workloads selected by --workloads={args.workloads!r}")
    rows: list[dict[str, Any]] = []
    for spec in specs:
        prompt = fit_prompt_to_token_budget(tokenizer, spec)
        actual_input_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        for batch_size in args.batch_sizes:
            for repeat_index in range(args.repeats):
                if spec.streaming:
                    row = measure_streaming(
                        args=args,
                        tokenizer=tokenizer,
                        model=model,
                        spec=spec,
                        prompt=prompt,
                        batch_size=batch_size,
                        repeat_index=repeat_index,
                    )
                else:
                    row = measure_nonstreaming(
                        args=args,
                        tokenizer=tokenizer,
                        model=model,
                        spec=spec,
                        prompt=prompt,
                        batch_size=batch_size,
                        repeat_index=repeat_index,
                    )
                row["model"] = args.model
                row["model_load_ms"] = model_load_ms
                row["actual_prompt_tokens_without_specials"] = actual_input_tokens
                rows.append(row)
    write_jsonl(out_dir / "overhead_measurements.jsonl", rows)
    write_json(out_dir / "overhead_summary.json", summarize_measurements(rows))
    return 0


def _measurement_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        return candidate / "overhead_measurements.jsonl"
    return candidate


def _summary_by_key(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    summary = summarize_measurements(rows)
    result = {}
    for group in summary["groups"]:
        key = (
            group.get("backend"),
            group.get("workload"),
            group.get("workload_family"),
            group.get("batch_size"),
            group.get("streaming"),
            group.get("target_input_tokens"),
            group.get("max_new_tokens"),
        )
        result[key] = group
    return result


def _metric_mean(group: dict[str, Any], metric: str) -> float | None:
    value = group.get("metrics", {}).get(metric, {}).get("mean")
    return float(value) if isinstance(value, (int, float)) else None


def _ratio(left: float | None, right: float | None) -> float | None:
    if left is None or right is None or right == 0:
        return None
    return left / right


def _find_baseline_key(
    keys: Iterable[tuple[Any, ...]],
    *,
    batch_size: int | None,
    baseline_family: str,
) -> tuple[Any, ...] | None:
    candidates = [
        key
        for key in keys
        if key[2] == baseline_family and key[4] is False and key[3] == batch_size
    ]
    if candidates:
        return sorted(candidates, key=lambda item: str(item))[0]
    candidates = [key for key in keys if key[2] == baseline_family and key[4] is False]
    return sorted(candidates, key=lambda item: str(item))[0] if candidates else None


def compare_runs(args: argparse.Namespace) -> int:
    out_dir = ensure_dir(args.out)
    cc_on_rows = read_jsonl(_measurement_path(args.cc_on))
    cc_off_rows = read_jsonl(_measurement_path(args.cc_off))
    cc_on = _summary_by_key(cc_on_rows)
    cc_off = _summary_by_key(cc_off_rows)
    common_keys = sorted(set(cc_on) & set(cc_off), key=lambda item: str(item))
    metrics = [
        "latency_ms",
        "ttft_ms",
        "ttft_proxy_ms",
        "tpot_ms",
        "tpot_excluding_ttft_proxy_ms",
        "tpot_after_first_chunk_ms",
        "input_h2d_ms",
        "tokens_per_second",
    ]

    comparisons = []
    for key in common_keys:
        row = group_key_dict(key)
        row["cc_on_rows"] = cc_on[key]["rows"]
        row["cc_off_rows"] = cc_off[key]["rows"]
        row["ratios"] = {}
        for metric in metrics:
            on_mean = _metric_mean(cc_on[key], metric)
            off_mean = _metric_mean(cc_off[key], metric)
            ratio = _ratio(on_mean, off_mean)
            row["ratios"][metric] = {
                "cc_on_mean": on_mean,
                "cc_off_mean": off_mean,
                "cc_on_over_cc_off": ratio,
            }
        baseline_key = _find_baseline_key(
            common_keys,
            batch_size=key[3],
            baseline_family=args.baseline_family,
        )
        row["ratio_of_ratios"] = {}
        if baseline_key is not None:
            row["baseline_workload"] = baseline_key[1]
            for metric in metrics:
                workload_ratio = row["ratios"][metric]["cc_on_over_cc_off"]
                baseline_ratio = _ratio(
                    _metric_mean(cc_on[baseline_key], metric),
                    _metric_mean(cc_off[baseline_key], metric),
                )
                row["ratio_of_ratios"][metric] = _ratio(
                    workload_ratio, baseline_ratio
                )
        comparisons.append(row)

    result = {
        "schema_version": COMPARISON_VERSION,
        "cc_on": str(_measurement_path(args.cc_on)),
        "cc_off": str(_measurement_path(args.cc_off)),
        "baseline_family": args.baseline_family,
        "common_groups": len(common_keys),
        "comparisons": comparisons,
    }
    write_json(out_dir / "overhead_comparison.json", result)
    write_comparison_csv(out_dir / "overhead_comparison.csv", comparisons, metrics)
    return 0


def write_comparison_csv(
    path: str | Path, comparisons: list[dict[str, Any]], metrics: list[str]
) -> None:
    columns = [
        "workload",
        "workload_family",
        "batch_size",
        "streaming",
        "target_input_tokens",
        "max_new_tokens",
        "cc_on_rows",
        "cc_off_rows",
    ]
    for metric in metrics:
        columns.extend(
            [
                f"{metric}_cc_on_mean",
                f"{metric}_cc_off_mean",
                f"{metric}_ratio",
                f"{metric}_ratio_of_ratios",
            ]
        )
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in comparisons:
            row = {column: item.get(column) for column in columns}
            for metric in metrics:
                values = item.get("ratios", {}).get(metric, {})
                row[f"{metric}_cc_on_mean"] = values.get("cc_on_mean")
                row[f"{metric}_cc_off_mean"] = values.get("cc_off_mean")
                row[f"{metric}_ratio"] = values.get("cc_on_over_cc_off")
                row[f"{metric}_ratio_of_ratios"] = item.get(
                    "ratio_of_ratios", {}
                ).get(metric)
            writer.writerow(row)


def list_workloads(args: argparse.Namespace) -> int:
    selected = (
        {item.strip() for item in args.workloads.split(",") if item.strip()}
        if args.workloads
        else None
    )
    specs = builtin_workloads(
        suite=args.suite,
        selected=selected,
        decode_output_tokens=args.decode_output_tokens,
        structured_output_tokens=args.structured_output_tokens,
        streaming_output_tokens=args.streaming_output_tokens,
        long_context_tokens=args.long_context_tokens,
    )
    for spec in specs:
        print(
            f"{spec.name}\tfamily={spec.family}\tmax_new_tokens={spec.max_new_tokens}"
            f"\tstreaming={spec.streaming}\ttarget_input_tokens={spec.target_input_tokens}"
        )
    return 0


def add_workload_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suite", choices=["smoke", "pilot", "full"], default="pilot")
    parser.add_argument(
        "--workloads",
        help=(
            "Comma-separated workload names or families. Families include "
            "baseline_chat, decode_heavy, structured_json_prompt, long_context, "
            "and streaming_steps."
        ),
    )
    parser.add_argument(
        "--decode-output-tokens", type=parse_int_list, default=None
    )
    parser.add_argument(
        "--structured-output-tokens", type=parse_int_list, default=None
    )
    parser.add_argument(
        "--streaming-output-tokens", type=parse_int_list, default=None
    )
    parser.add_argument("--long-context-tokens", type=parse_int_list, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run prompt-induced GPU CC overhead amplification experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run-transformers",
        help="Run built-in workloads through local Transformers generate().",
    )
    run.add_argument("--out", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--run-id", default=str(uuid.uuid4()))
    run.add_argument("--cc-mode", choices=["on", "off", "unknown"], default="unknown")
    run.add_argument("--batch-sizes", type=parse_int_list, default=parse_int_list("1"))
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--warmup", type=int, default=1)
    run.add_argument("--dtype", choices=["bf16", "auto"], default="bf16")
    run.add_argument("--device-map", default="auto")
    run.add_argument("--trust-remote-code", action="store_true")
    run.add_argument("--measure-ttft-proxy", action="store_true")
    run.add_argument("--do-sample", action="store_true")
    add_workload_args(run)
    run.set_defaults(func=run_transformers)

    compare = subparsers.add_parser(
        "compare",
        help="Compare one CC-On run and one CC-Off run.",
    )
    compare.add_argument("--cc-on", required=True, help="Run directory or JSONL file.")
    compare.add_argument("--cc-off", required=True, help="Run directory or JSONL file.")
    compare.add_argument("--out", required=True)
    compare.add_argument("--baseline-family", default="baseline_chat")
    compare.set_defaults(func=compare_runs)

    listed = subparsers.add_parser("list-workloads")
    add_workload_args(listed)
    listed.set_defaults(func=list_workloads)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
