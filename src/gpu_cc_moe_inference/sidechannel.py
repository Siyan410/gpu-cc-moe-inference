from __future__ import annotations

import argparse
import math
import random
import re
import statistics
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .env import write_env
from .io import ensure_dir, read_jsonl, write_json, write_jsonl
from .prompting import load_prompts
from .schema import validate_attacker_feature_row, validate_routing_label_row
from .torch_utils import import_torch, require_cuda


LABEL_SCHEMA_VERSION = "routing-label-v1"
FEATURE_SCHEMA_VERSION = "attacker-feature-v1"


def numeric_feature_vector(rows: list[dict[str, Any]]) -> tuple[list[str], list[list[float]]]:
    keys: list[str] = []
    for row in rows:
        for key, value in row.get("features", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and key not in keys:
                keys.append(key)
    vectors: list[list[float]] = []
    for row in rows:
        features = row.get("features", {})
        vectors.append([float(features.get(key, 0.0)) for key in keys])
    return keys, vectors


def standardize(train: list[list[float]], test: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    if not train:
        return train, test
    width = len(train[0])
    means = [statistics.fmean(row[index] for row in train) for index in range(width)]
    stdevs = []
    for index in range(width):
        values = [row[index] for row in train]
        stdev = statistics.pstdev(values)
        stdevs.append(stdev if stdev > 1e-12 else 1.0)

    def transform(rows: list[list[float]]) -> list[list[float]]:
        return [
            [(row[index] - means[index]) / stdevs[index] for index in range(width)]
            for row in rows
        ]

    return transform(train), transform(test)


def squared_distance(left: list[float], right: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def label_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row["prompt_id"]), int(row["token_index"]), int(row.get("layer", 0)))


def feature_key(row: dict[str, Any], layer: int) -> tuple[str, int, int]:
    return (str(row["prompt_id"]), int(row["token_index"]), layer)


def join_labels_features(
    labels: list[dict[str, Any]], features: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    layers = sorted({int(row["layer"]) for row in labels})
    feature_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    layerless_feature_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for feature in features:
        prompt_id = str(feature["prompt_id"])
        token_index = int(feature["token_index"])
        if "layer" in feature:
            feature_by_key[(prompt_id, token_index, int(feature["layer"]))] = feature
        layerless_feature_by_key[(prompt_id, token_index)] = feature
    joined = []
    for label in labels:
        key = label_key(label)
        feature = feature_by_key.get(key) or layerless_feature_by_key.get((key[0], key[1]))
        if feature is None:
            continue
        row = {
            "prompt_id": key[0],
            "token_index": key[1],
            "layer": key[2],
            "topk": [int(item) for item in label["topk"]],
            "features": dict(feature.get("features", {})),
        }
        row["features"].setdefault("layer", float(key[2]))
        row["features"].setdefault("token_index", float(key[1]))
        joined.append(row)
    if layers and not joined:
        raise ValueError("no labels could be joined to attacker features")
    return joined


def split_train_test(rows: list[dict[str, Any]], test_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    prompt_ids = sorted({str(row["prompt_id"]) for row in rows})
    rng = random.Random(seed)
    rng.shuffle(prompt_ids)
    test_count = max(1, int(round(len(prompt_ids) * test_fraction))) if len(prompt_ids) > 1 else 1
    test_prompts = set(prompt_ids[:test_count])
    train_indices = [index for index, row in enumerate(rows) if str(row["prompt_id"]) not in test_prompts]
    test_indices = [index for index, row in enumerate(rows) if str(row["prompt_id"]) in test_prompts]
    if not train_indices:
        midpoint = max(1, len(rows) // 2)
        train_indices = list(range(midpoint))
        test_indices = list(range(midpoint, len(rows))) or list(range(midpoint))
    return train_indices, test_indices


def top_scores_from_neighbors(
    train_vectors: list[list[float]],
    train_labels: list[list[int]],
    query: list[float],
    neighbors: int,
) -> dict[int, float]:
    distances = [
        (squared_distance(query, vector), index) for index, vector in enumerate(train_vectors)
    ]
    distances.sort(key=lambda item: item[0])
    chosen = distances[: max(1, min(neighbors, len(distances)))]
    scores: dict[int, float] = defaultdict(float)
    total_weight = 0.0
    for distance, index in chosen:
        weight = 1.0 / (math.sqrt(distance) + 1e-6)
        total_weight += weight
        for expert in train_labels[index]:
            scores[expert] += weight
    if total_weight > 0:
        for expert in list(scores):
            scores[expert] /= total_weight
    return dict(scores)


def predict_knn(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    neighbors: int,
) -> list[dict[str, Any]]:
    feature_keys, all_vectors = numeric_feature_vector(train_rows + test_rows)
    train_vectors = all_vectors[: len(train_rows)]
    test_vectors = all_vectors[len(train_rows) :]
    train_vectors, test_vectors = standardize(train_vectors, test_vectors)
    train_labels = [[int(item) for item in row["topk"]] for row in train_rows]
    all_experts = sorted({expert for labels in train_labels for expert in labels})
    if not all_experts:
        raise ValueError("training rows do not contain expert labels")
    predictions = []
    default_k = len(train_labels[0])
    for row, vector in zip(test_rows, test_vectors):
        scores = top_scores_from_neighbors(train_vectors, train_labels, vector, neighbors)
        for expert in all_experts:
            scores.setdefault(expert, 0.0)
        k = len(row["topk"]) or default_k
        predicted = [
            expert
            for expert, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        ]
        predictions.append(
            {
                "prompt_id": row["prompt_id"],
                "token_index": row["token_index"],
                "layer": row["layer"],
                "true": [int(item) for item in row["topk"]],
                "predicted": predicted,
                "scores": scores,
                "feature_keys": feature_keys,
            }
        )
    return predictions


def exact_match(true: list[int], predicted: list[int]) -> float:
    return 1.0 if set(true) == set(predicted) else 0.0


def jaccard(true: list[int], predicted: list[int]) -> float:
    left = set(true)
    right = set(predicted)
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def f1_score(true: list[int], predicted: list[int]) -> float:
    left = set(true)
    right = set(predicted)
    tp = len(left & right)
    fp = len(right - left)
    fn = len(left - right)
    if tp == fp == fn == 0:
        return 1.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def auc_rank(labels: list[int], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = len(positives) * len(negatives)
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / total


def summarize_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        raise ValueError("no predictions to summarize")
    expert_ids = sorted(
        {
            expert
            for pred in predictions
            for expert in list(pred["true"]) + list(pred["predicted"]) + list(pred["scores"].keys())
        }
    )
    per_expert_auc: dict[str, float] = {}
    per_expert_f1: dict[str, float] = {}
    for expert in expert_ids:
        labels = [1 if expert in set(pred["true"]) else 0 for pred in predictions]
        scores = [float(pred["scores"].get(expert, 0.0)) for pred in predictions]
        auc = auc_rank(labels, scores)
        if auc is not None:
            per_expert_auc[str(expert)] = auc
        predicted_labels = [
            1 if expert in set(pred["predicted"]) else 0 for pred in predictions
        ]
        tp = sum(1 for truth, pred in zip(labels, predicted_labels) if truth == pred == 1)
        fp = sum(1 for truth, pred in zip(labels, predicted_labels) if truth == 0 and pred == 1)
        fn = sum(1 for truth, pred in zip(labels, predicted_labels) if truth == 1 and pred == 0)
        if tp or fp or fn:
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            per_expert_f1[str(expert)] = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        by_layer[str(pred["layer"])].append(pred)
    return {
        "rows": len(predictions),
        "topk_exact_match": statistics.fmean(
            exact_match(pred["true"], pred["predicted"]) for pred in predictions
        ),
        "topk_jaccard": statistics.fmean(
            jaccard(pred["true"], pred["predicted"]) for pred in predictions
        ),
        "micro_f1": statistics.fmean(
            f1_score(pred["true"], pred["predicted"]) for pred in predictions
        ),
        "per_expert_auc_macro": statistics.fmean(per_expert_auc.values())
        if per_expert_auc
        else None,
        "per_expert_auc": per_expert_auc,
        "per_expert_f1_macro": statistics.fmean(per_expert_f1.values())
        if per_expert_f1
        else None,
        "per_expert_f1": per_expert_f1,
        "per_layer": {
            layer: {
                "rows": len(items),
                "topk_exact_match": statistics.fmean(
                    exact_match(pred["true"], pred["predicted"]) for pred in items
                ),
                "topk_jaccard": statistics.fmean(
                    jaccard(pred["true"], pred["predicted"]) for pred in items
                ),
                "micro_f1": statistics.fmean(
                    f1_score(pred["true"], pred["predicted"]) for pred in items
                ),
            }
            for layer, items in sorted(by_layer.items(), key=lambda item: int(item[0]))
        },
    }


def evaluate_sidechannel(
    labels: list[dict[str, Any]],
    features: list[dict[str, Any]],
    *,
    seed: int = 7,
    test_fraction: float = 0.3,
    neighbors: int = 5,
) -> dict[str, Any]:
    for row in labels:
        validate_routing_label_row(row)
    for row in features:
        validate_attacker_feature_row(row)
    joined = join_labels_features(labels, features)
    train_indices, test_indices = split_train_test(joined, test_fraction, seed)
    train_rows = [joined[index] for index in train_indices]
    test_rows = [joined[index] for index in test_indices]
    predictions = predict_knn(train_rows, test_rows, neighbors=neighbors)

    rng = random.Random(seed)
    shuffled_train = [dict(row) for row in train_rows]
    shuffled_labels = [row["topk"] for row in shuffled_train]
    rng.shuffle(shuffled_labels)
    for row, shuffled in zip(shuffled_train, shuffled_labels):
        row["topk"] = shuffled
    shuffle_predictions = predict_knn(shuffled_train, test_rows, neighbors=neighbors)

    length_predictions = predict_length_matched(train_rows, test_rows)
    return {
        "schema_version": "sidechannel-eval-v1",
        "rows_joined": len(joined),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "model": "standardized_knn_multilabel",
        "neighbors": neighbors,
        "metrics": summarize_predictions(predictions),
        "negative_controls": {
            "label_shuffle": summarize_predictions(shuffle_predictions),
            "length_matched_prompt": summarize_predictions(length_predictions),
        },
        "interpretation_note": (
            "Labels are offline ground truth only. Attacker features must not include "
            "router logits, top-k tensors, or hidden states."
        ),
    }


def predict_length_matched(
    train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    predictions = []
    all_labels = [label for row in train_rows for label in row["topk"]]
    fallback = [expert for expert, _ in Counter(all_labels).most_common(len(train_rows[0]["topk"]))]
    for test in test_rows:
        target_len = float(test.get("features", {}).get("prompt_chars", 0.0))
        ranked_train = sorted(
            train_rows,
            key=lambda row: abs(float(row.get("features", {}).get("prompt_chars", 0.0)) - target_len),
        )
        counter: Counter[int] = Counter()
        for row in ranked_train[: max(1, min(5, len(ranked_train)))]:
            counter.update(row["topk"])
        predicted = [expert for expert, _ in counter.most_common(len(test["topk"]))]
        if len(predicted) < len(test["topk"]):
            predicted.extend(expert for expert in fallback if expert not in predicted)
        predictions.append(
            {
                "prompt_id": test["prompt_id"],
                "token_index": test["token_index"],
                "layer": test["layer"],
                "true": test["topk"],
                "predicted": predicted[: len(test["topk"])],
                "scores": {expert: float(score) for expert, score in counter.items()},
            }
        )
    return predictions


def trusted_label_run(args: argparse.Namespace) -> int:
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
    pattern = re.compile(args.router_module_regex)
    captured: list[dict[str, Any]] = []

    def make_hook(name: str):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if not hasattr(tensor, "detach") or tensor.ndim < 2:
                return
            logits = tensor.detach()
            if logits.ndim == 3:
                logits = logits.reshape(-1, logits.shape[-1])
            if logits.shape[-1] < args.top_k:
                return
            topk = torch.topk(logits.float(), k=args.top_k, dim=-1).indices.cpu().tolist()
            layer_match = re.search(r"layers?\.(\d+)", name)
            layer = int(layer_match.group(1)) if layer_match else -1
            captured.append({"module": name, "layer": layer, "topk_rows": topk})

        return hook

    hooks = []
    for name, module in model.named_modules():
        if pattern.search(name):
            hooks.append(module.register_forward_hook(make_hook(name)))
    if not hooks:
        raise RuntimeError(f"no router modules matched regex: {args.router_module_regex}")

    rows: list[dict[str, Any]] = []
    try:
        with torch.inference_mode():
            for prompt_id, prompt in enumerate(prompts):
                captured.clear()
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                model(**inputs)
                for capture in captured:
                    for token_index, experts in enumerate(capture["topk_rows"]):
                        row = {
                            "schema_version": LABEL_SCHEMA_VERSION,
                            "run_id": args.run_id,
                            "prompt_id": str(prompt_id),
                            "prompt_chars": len(prompt),
                            "layer": int(capture["layer"]),
                            "token_index": token_index,
                            "topk": [int(item) for item in experts],
                            "source": "trusted_router_hook_offline_ground_truth",
                            "module": capture["module"],
                        }
                        validate_routing_label_row(row)
                        rows.append(row)
    finally:
        for hook in hooks:
            hook.remove()
    write_jsonl(out_dir / "routing_labels.jsonl", rows)
    return 0


def attacker_feature_run(args: argparse.Namespace) -> int:
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
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for prompt_id, prompt in enumerate(prompts):
            tokenization_start = time.perf_counter()
            encoded = tokenizer(prompt, return_tensors="pt")
            tokenization_ms = (time.perf_counter() - tokenization_start) * 1000
            input_ids = encoded["input_ids"]
            input_bytes = input_ids.numel() * input_ids.element_size()
            h2d_start = time.perf_counter()
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            torch.cuda.synchronize()
            h2d_ms = (time.perf_counter() - h2d_start) * 1000
            generate_start = time.perf_counter()
            output = model.generate(**encoded, max_new_tokens=args.max_new_tokens, do_sample=False)
            torch.cuda.synchronize()
            generate_ms = (time.perf_counter() - generate_start) * 1000
            decode_start = time.perf_counter()
            tokenizer.batch_decode(output, skip_special_tokens=True)
            decode_output_ms = (time.perf_counter() - decode_start) * 1000
            output_bytes = output.numel() * output.element_size()
            input_tokens = int(input_ids.shape[-1])
            output_tokens = int(output.shape[-1])
            for token_index in range(input_tokens):
                row = {
                    "schema_version": FEATURE_SCHEMA_VERSION,
                    "run_id": args.run_id,
                    "prompt_id": str(prompt_id),
                    "token_index": token_index,
                    "attacker_policy": "no_router_logits_no_topk_no_hidden_states",
                    "features": {
                        "prompt_chars": float(len(prompt)),
                        "input_tokens": float(input_tokens),
                        "output_tokens": float(output_tokens),
                        "tokenization_ms": tokenization_ms,
                        "input_h2d_bytes": float(input_bytes),
                        "input_h2d_ms": h2d_ms,
                        "generate_cpu_wall_ms": generate_ms,
                        "decode_output_ms": decode_output_ms,
                        "output_d2h_bytes": float(output_bytes),
                    },
                }
                validate_attacker_feature_row(row)
                rows.append(row)
    write_jsonl(out_dir / "attacker_features.jsonl", rows)
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    out_dir = ensure_dir(args.out)
    write_env(out_dir)
    labels = read_jsonl(args.labels)
    features = read_jsonl(args.features)
    result = evaluate_sidechannel(
        labels,
        features,
        seed=args.seed,
        test_fraction=args.test_fraction,
        neighbors=args.neighbors,
    )
    write_json(out_dir / "sidechannel_eval.json", result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run routing side-channel experiments.")
    parser.add_argument("--run-id", default=str(uuid.uuid4()))
    subparsers = parser.add_subparsers(dest="command", required=True)

    trusted = subparsers.add_parser("trusted-label-run")
    trusted.add_argument("--out", required=True)
    trusted.add_argument("--model", required=True)
    trusted.add_argument("--prompts")
    trusted.add_argument("--top-k", type=int, default=8)
    trusted.add_argument("--dtype", choices=["bf16", "auto"], default="bf16")
    trusted.add_argument("--device-map", default="auto")
    trusted.add_argument("--trust-remote-code", action="store_true")
    trusted.add_argument("--router-module-regex", default=r"(^|\.)(router|gate)$")
    trusted.set_defaults(func=trusted_label_run)

    attacker = subparsers.add_parser("attacker-feature-run")
    attacker.add_argument("--out", required=True)
    attacker.add_argument("--model", required=True)
    attacker.add_argument("--prompts")
    attacker.add_argument("--max-new-tokens", type=int, default=16)
    attacker.add_argument("--dtype", choices=["bf16", "auto"], default="bf16")
    attacker.add_argument("--device-map", default="auto")
    attacker.add_argument("--trust-remote-code", action="store_true")
    attacker.set_defaults(func=attacker_feature_run)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--labels", required=True)
    evaluate.add_argument("--features", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--seed", type=int, default=7)
    evaluate.add_argument("--test-fraction", type=float, default=0.3)
    evaluate.add_argument("--neighbors", type=int, default=5)
    evaluate.set_defaults(func=evaluate_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
