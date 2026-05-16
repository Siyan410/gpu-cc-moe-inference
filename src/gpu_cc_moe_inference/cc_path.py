from __future__ import annotations

import argparse
import math
import resource
import time
import uuid
from pathlib import Path
from typing import Any

from .env import write_env
from .io import ensure_dir, write_jsonl
from .schema import validate_cc_path_row
from .torch_utils import require_cuda


SCHEMA_VERSION = "cc-path-v1"


def parse_size_bytes(value: str) -> int:
    raw = value.strip().lower()
    scale = 1
    for suffix, multiplier in (
        ("gib", 1024**3),
        ("gb", 1000**3),
        ("mib", 1024**2),
        ("mb", 1000**2),
        ("kib", 1024),
        ("kb", 1000),
        ("b", 1),
    ):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
            scale = multiplier
            break
    return int(float(raw) * scale)


def parse_sizes(values: str) -> list[int]:
    sizes = [parse_size_bytes(item) for item in values.split(",") if item.strip()]
    if not sizes:
        raise argparse.ArgumentTypeError("at least one transfer size is required")
    return sizes


def stage_estimates(
    *,
    bytes_count: int,
    cpu_wall_ms: float,
    cuda_event_ms: float,
    driver_call_ms: float,
    effective_bandwidth_gbps: float,
) -> dict[str, Any]:
    theoretical_transfer_ms = (bytes_count / (effective_bandwidth_gbps * 1_000_000_000)) * 1000
    cpu_to_bounce = min(max(driver_call_ms, 0.0), theoretical_transfer_ms)
    cpu_encrypt = max(0.0, driver_call_ms - cpu_to_bounce)
    bounce_to_gpu = max(0.0, cuda_event_ms)
    residual = max(0.0, cpu_wall_ms - driver_call_ms - bounce_to_gpu)
    return {
        "cpu_stage_encrypt_est_ms": cpu_encrypt,
        "cpu_to_bounce_est_ms": cpu_to_bounce,
        "bounce_to_gpu_dma_est_ms": bounce_to_gpu,
        "gpu_decrypt_residual_est_ms": residual,
        "estimate_provenance": {
            "cpu_stage_encrypt_est_ms": {
                "estimated": True,
                "method": "residual_from_driver_call_minus_theoretical_cpu_to_bounce",
            },
            "cpu_to_bounce_est_ms": {
                "estimated": True,
                "method": f"min(driver_call_ms, bytes/{effective_bandwidth_gbps}GBps)",
            },
            "bounce_to_gpu_dma_est_ms": {
                "estimated": True,
                "method": "cuda_event_elapsed_time_proxy_not_internal_dma_observation",
            },
            "gpu_decrypt_residual_est_ms": {
                "estimated": True,
                "method": "max(0, cpu_wall_ms-driver_call_ms-cuda_event_ms)",
            },
            "direct_measurements": [
                "cpu_wall_ms",
                "cuda_event_ms",
                "driver_call_ms",
                "sync_wait_ms",
                "driver_syscall_cpu_ms",
            ],
        },
    }


def _resource_system_ms() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_stime * 1000


def measure_copy(
    *,
    torch: Any,
    direction: str,
    bytes_count: int,
    repeat_index: int,
    run_id: str,
    effective_bandwidth_gbps: float,
    pin_memory: bool,
) -> dict[str, Any]:
    cpu_tensor = torch.empty(bytes_count, dtype=torch.uint8, pin_memory=pin_memory)
    gpu_tensor = torch.empty(bytes_count, dtype=torch.uint8, device="cuda")
    if direction == "H2D":
        cpu_tensor.fill_(repeat_index % 251)
    else:
        gpu_tensor.fill_(repeat_index % 251)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_sys_ms = _resource_system_ms()
    start_wall = time.perf_counter()
    start_event.record()
    before_driver = time.perf_counter()
    if direction == "H2D":
        gpu_tensor.copy_(cpu_tensor, non_blocking=pin_memory)
    elif direction == "D2H":
        cpu_tensor.copy_(gpu_tensor, non_blocking=pin_memory)
    else:
        raise ValueError(f"unsupported direction {direction}")
    after_driver = time.perf_counter()
    end_event.record()
    torch.cuda.synchronize()
    end_wall = time.perf_counter()
    end_sys_ms = _resource_system_ms()

    driver_call_ms = (after_driver - before_driver) * 1000
    cpu_wall_ms = (end_wall - start_wall) * 1000
    cuda_event_ms = float(start_event.elapsed_time(end_event))
    sync_wait_ms = max(0.0, cpu_wall_ms - driver_call_ms)
    row = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "repeat_index": repeat_index,
        "direction": direction,
        "bytes": bytes_count,
        "pin_memory": pin_memory,
        "cpu_wall_ms": cpu_wall_ms,
        "cuda_event_ms": cuda_event_ms,
        "sync_wait_ms": sync_wait_ms,
        "driver_call_ms": driver_call_ms,
        "driver_syscall_cpu_ms": max(0.0, end_sys_ms - start_sys_ms),
    }
    row.update(
        stage_estimates(
            bytes_count=bytes_count,
            cpu_wall_ms=cpu_wall_ms,
            cuda_event_ms=cuda_event_ms,
            driver_call_ms=driver_call_ms,
            effective_bandwidth_gbps=effective_bandwidth_gbps,
        )
    )
    validate_cc_path_row(row)
    return row


def run_transfer_bench(args: argparse.Namespace) -> list[dict[str, Any]]:
    torch = require_cuda()
    rows: list[dict[str, Any]] = []
    for bytes_count in args.sizes:
        for direction in args.directions:
            for index in range(args.warmup):
                measure_copy(
                    torch=torch,
                    direction=direction,
                    bytes_count=bytes_count,
                    repeat_index=-index - 1,
                    run_id=args.run_id,
                    effective_bandwidth_gbps=args.effective_bandwidth_gbps,
                    pin_memory=args.pin_memory,
                )
            for index in range(args.repeat):
                rows.append(
                    measure_copy(
                        torch=torch,
                        direction=direction,
                        bytes_count=bytes_count,
                        repeat_index=index,
                        run_id=args.run_id,
                        effective_bandwidth_gbps=args.effective_bandwidth_gbps,
                        pin_memory=args.pin_memory,
                    )
                )
    return rows


def qwen3_moe_gemm_specs(
    tokens_per_expert: list[int],
    hidden_size: int = 2048,
    moe_intermediate_size: int = 768,
    num_experts: int = 128,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for tokens in tokens_per_expert:
        specs.extend(
            [
                {
                    "operator": "expert_gate_projection",
                    "tokens_per_expert": tokens,
                    "m": tokens,
                    "k": hidden_size,
                    "n": moe_intermediate_size,
                    "dtype": "bfloat16",
                    "num_experts": num_experts,
                },
                {
                    "operator": "expert_up_projection",
                    "tokens_per_expert": tokens,
                    "m": tokens,
                    "k": hidden_size,
                    "n": moe_intermediate_size,
                    "dtype": "bfloat16",
                    "num_experts": num_experts,
                },
                {
                    "operator": "expert_down_projection",
                    "tokens_per_expert": tokens,
                    "m": tokens,
                    "k": moe_intermediate_size,
                    "n": hidden_size,
                    "dtype": "bfloat16",
                    "num_experts": num_experts,
                },
            ]
        )
    return specs


def gemm_flops(m: int, k: int, n: int) -> int:
    return 2 * m * k * n


def theoretical_ms_from_peak(flops: float, peak_tflops: float) -> float:
    if peak_tflops <= 0:
        raise ValueError("peak_tflops must be positive")
    return (flops / (peak_tflops * 1_000_000_000_000)) * 1000


def measure_gemm_roofline(
    *,
    torch: Any,
    spec: dict[str, Any],
    repeat: int,
    warmup: int,
    peak_tflops: float,
) -> dict[str, Any]:
    a = torch.randn((spec["m"], spec["k"]), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((spec["k"], spec["n"]), device="cuda", dtype=torch.bfloat16)
    for _ in range(warmup):
        torch.mm(a, b)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        torch.mm(a, b)
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = float(start.elapsed_time(end)) / repeat
    flops = gemm_flops(spec["m"], spec["k"], spec["n"])
    achieved_tflops = (flops / (elapsed_ms / 1000)) / 1_000_000_000_000 if elapsed_ms > 0 else math.inf
    row = dict(spec)
    row.update(
        {
            "kind": "compute",
            "flops": flops,
            "measured_ms": elapsed_ms,
            "theoretical_ms": theoretical_ms_from_peak(flops, peak_tflops),
            "achieved_tflops": achieved_tflops,
            "peak_tflops_reference": peak_tflops,
        }
    )
    return row


def transfer_roofline_rows(
    *,
    sizes: list[int],
    effective_bandwidth_gbps: float,
) -> list[dict[str, Any]]:
    rows = []
    for bytes_count in sizes:
        rows.append(
            {
                "kind": "io",
                "operator": "host_device_tensor_transfer",
                "bytes": bytes_count,
                "theoretical_ms": (bytes_count / (effective_bandwidth_gbps * 1_000_000_000)) * 1000,
                "effective_bandwidth_gbps_reference": effective_bandwidth_gbps,
            }
        )
    return rows


def moe_io_roofline_rows(
    *,
    token_counts: list[int],
    effective_bandwidth_gbps: float,
    hidden_size: int = 2048,
    num_experts_per_tok: int = 8,
    activation_bytes: int = 2,
    token_id_bytes: int = 8,
    expert_index_bytes: int = 8,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tokens in token_counts:
        specs = [
            {
                "operator": "token_ids_h2d",
                "direction": "H2D",
                "bytes": tokens * token_id_bytes,
                "description": "input token id tensor",
            },
            {
                "operator": "hidden_states_h2d",
                "direction": "H2D",
                "bytes": tokens * hidden_size * activation_bytes,
                "description": "BF16 hidden-state tensor shape [tokens, hidden_size]",
            },
            {
                "operator": "router_like_topk_index_tensor_h2d",
                "direction": "H2D",
                "bytes": tokens * num_experts_per_tok * expert_index_bytes,
                "description": "top-k expert index tensor shape [tokens, experts_per_token]",
            },
            {
                "operator": "expert_dispatch_like_gather_buffer_h2d",
                "direction": "H2D",
                "bytes": tokens * num_experts_per_tok * hidden_size * activation_bytes,
                "description": "BF16 expert dispatch gather/scatter buffer",
            },
            {
                "operator": "expert_dispatch_like_scatter_buffer_d2h",
                "direction": "D2H",
                "bytes": tokens * num_experts_per_tok * hidden_size * activation_bytes,
                "description": "BF16 expert dispatch gather/scatter buffer",
            },
        ]
        for spec in specs:
            row = {
                "kind": "io_moe_shape",
                "tokens": tokens,
                "hidden_size": hidden_size,
                "num_experts_per_tok": num_experts_per_tok,
                **spec,
            }
            row["theoretical_ms"] = (
                row["bytes"] / (effective_bandwidth_gbps * 1_000_000_000)
            ) * 1000
            row["effective_bandwidth_gbps_reference"] = effective_bandwidth_gbps
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    lines = [",".join(columns)]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            text = str(value).replace('"', '""')
            if "," in text or "\n" in text:
                text = f'"{text}"'
            values.append(text)
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_operator_roofline(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = transfer_roofline_rows(
        sizes=args.sizes, effective_bandwidth_gbps=args.effective_bandwidth_gbps
    )
    io_tokens = [int(item) for item in args.io_token_counts.split(",") if item.strip()]
    rows.extend(
        moe_io_roofline_rows(
            token_counts=io_tokens,
            effective_bandwidth_gbps=args.effective_bandwidth_gbps,
            hidden_size=args.hidden_size,
            num_experts_per_tok=args.num_experts_per_tok,
        )
    )
    torch = require_cuda()
    tokens = [int(item) for item in args.tokens_per_expert.split(",") if item.strip()]
    for spec in qwen3_moe_gemm_specs(
        tokens,
        hidden_size=args.hidden_size,
        moe_intermediate_size=args.moe_intermediate_size,
        num_experts=args.num_experts,
    ):
        rows.append(
            measure_gemm_roofline(
                torch=torch,
                spec=spec,
                repeat=args.gemm_repeat,
                warmup=args.warmup,
                peak_tflops=args.h20_bf16_peak_tflops,
            )
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure CPU TEE to GPU TEE transfer path costs under GPU CC."
    )
    parser.add_argument("--out", required=True, help="Output run directory.")
    parser.add_argument("--run-id", default=None, help="Stable run id. Defaults to a UUID.")
    parser.add_argument(
        "--sizes",
        type=parse_sizes,
        default=parse_sizes("4KB,64KB,1MB,16MB,64MB"),
        help="Comma-separated transfer sizes, e.g. 4KB,1MB,64MB.",
    )
    parser.add_argument("--directions", nargs="+", choices=["H2D", "D2H"], default=["H2D", "D2H"])
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--effective-bandwidth-gbps",
        type=float,
        default=4.0,
        help="Reference CC interconnect bandwidth used only for estimates/theory.",
    )
    parser.add_argument(
        "--operator-roofline",
        action="store_true",
        help="Also write operator_roofline.csv for transfer and Qwen3 MoE GEMM shapes.",
    )
    parser.add_argument("--tokens-per-expert", default="1,4,16,64,256")
    parser.add_argument("--io-token-counts", default="1,16,128,1024")
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--moe-intermediate-size", type=int, default=768)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--num-experts-per-tok", type=int, default=8)
    parser.add_argument("--gemm-repeat", type=int, default=50)
    parser.add_argument(
        "--h20-bf16-peak-tflops",
        type=float,
        default=148.0,
        help="Reference H20 BF16 peak used for theory; override with calibrated value.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.run_id = args.run_id or str(uuid.uuid4())
    out_dir = ensure_dir(args.out)
    write_env(out_dir)
    rows = run_transfer_bench(args)
    write_jsonl(out_dir / "cc_path_measurements.jsonl", rows)
    if args.operator_roofline:
        roofline_rows = run_operator_roofline(args)
        write_csv(out_dir / "operator_roofline.csv", roofline_rows)
    return 0
