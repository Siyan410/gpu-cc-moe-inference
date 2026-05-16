from __future__ import annotations

from typing import Any


CC_PATH_REQUIRED = {
    "schema_version",
    "run_id",
    "direction",
    "bytes",
    "cpu_wall_ms",
    "cuda_event_ms",
    "sync_wait_ms",
    "driver_call_ms",
    "driver_syscall_cpu_ms",
    "cpu_stage_encrypt_est_ms",
    "cpu_to_bounce_est_ms",
    "bounce_to_gpu_dma_est_ms",
    "gpu_decrypt_residual_est_ms",
    "estimate_provenance",
}

ROUTING_LABEL_REQUIRED = {
    "schema_version",
    "run_id",
    "prompt_id",
    "layer",
    "token_index",
    "topk",
}

ATTACKER_FEATURE_REQUIRED = {
    "schema_version",
    "run_id",
    "prompt_id",
    "token_index",
    "features",
}


def require_fields(row: dict[str, Any], required: set[str], row_name: str) -> None:
    missing = sorted(required.difference(row))
    if missing:
        raise ValueError(f"{row_name} missing required fields: {', '.join(missing)}")


def validate_cc_path_row(row: dict[str, Any]) -> None:
    require_fields(row, CC_PATH_REQUIRED, "cc_path_measurement")
    if row["direction"] not in {"H2D", "D2H"}:
        raise ValueError("direction must be H2D or D2H")
    if int(row["bytes"]) <= 0:
        raise ValueError("bytes must be positive")
    if not row["estimate_provenance"].get("bounce_to_gpu_dma_est_ms", {}).get("estimated"):
        raise ValueError("bounce_to_gpu_dma_est_ms must be marked estimated")


def validate_routing_label_row(row: dict[str, Any]) -> None:
    require_fields(row, ROUTING_LABEL_REQUIRED, "routing_label")
    if not isinstance(row["topk"], list) or not row["topk"]:
        raise ValueError("topk must be a non-empty list")


def validate_attacker_feature_row(row: dict[str, Any]) -> None:
    require_fields(row, ATTACKER_FEATURE_REQUIRED, "attacker_feature")
    if not isinstance(row["features"], dict):
        raise ValueError("features must be an object")
