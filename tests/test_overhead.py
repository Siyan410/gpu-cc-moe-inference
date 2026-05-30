import argparse
import json
import tempfile
import unittest
from pathlib import Path

from gpu_cc_moe_inference.io import write_jsonl
from gpu_cc_moe_inference.overhead import (
    builtin_workloads,
    compare_runs,
    metric_summary,
    parse_int_list,
    percentile,
    summarize_measurements,
)


class OverheadTests(unittest.TestCase):
    def test_parse_int_list(self):
        self.assertEqual(parse_int_list("1,8,32"), [1, 8, 32])
        with self.assertRaises(Exception):
            parse_int_list("0")

    def test_percentile_interpolates(self):
        self.assertEqual(percentile([1, 2, 3, 4], 0.5), 2.5)
        self.assertEqual(percentile([10], 0.99), 10.0)
        self.assertIsNone(percentile([], 0.95))

    def test_metric_summary_ignores_nonfinite(self):
        summary = metric_summary([1.0, 2.0, float("nan")])
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["mean"], 1.5)

    def test_builtin_workload_selection_by_family(self):
        specs = builtin_workloads(suite="smoke", selected={"decode_heavy"})
        self.assertEqual([spec.family for spec in specs], ["decode_heavy"])
        self.assertEqual(specs[0].max_new_tokens, 64)

    def test_summarize_measurements_groups_ok_rows(self):
        rows = [
            {
                "status": "ok",
                "backend": "transformers",
                "workload": "baseline_chat_out32",
                "workload_family": "baseline_chat",
                "batch_size": 1,
                "streaming": False,
                "target_input_tokens": None,
                "max_new_tokens": 32,
                "latency_ms": 10.0,
                "tpot_ms": 1.0,
            },
            {
                "status": "skipped",
                "backend": "transformers",
                "workload": "baseline_chat_out32",
                "workload_family": "baseline_chat",
                "batch_size": 1,
                "streaming": False,
                "target_input_tokens": None,
                "max_new_tokens": 32,
                "latency_ms": 100.0,
            },
        ]
        summary = summarize_measurements(rows)
        self.assertEqual(summary["rows"], 1)
        self.assertEqual(summary["groups"][0]["metrics"]["latency_ms"]["mean"], 10.0)

    def test_compare_runs_computes_ratio_of_ratios(self):
        def row(workload, family, latency):
            return {
                "status": "ok",
                "backend": "transformers",
                "workload": workload,
                "workload_family": family,
                "batch_size": 1,
                "streaming": False,
                "target_input_tokens": None,
                "max_new_tokens": 32,
                "latency_ms": latency,
                "tpot_ms": latency / 32,
            }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            on = tmp_path / "on.jsonl"
            off = tmp_path / "off.jsonl"
            out = tmp_path / "out"
            write_jsonl(
                on,
                [
                    row("baseline_chat_out32", "baseline_chat", 20.0),
                    row("decode_heavy_out32", "decode_heavy", 60.0),
                ],
            )
            write_jsonl(
                off,
                [
                    row("baseline_chat_out32", "baseline_chat", 10.0),
                    row("decode_heavy_out32", "decode_heavy", 20.0),
                ],
            )
            compare_runs(
                argparse.Namespace(
                    cc_on=str(on),
                    cc_off=str(off),
                    out=str(out),
                    baseline_family="baseline_chat",
                )
            )
            result = json.loads((out / "overhead_comparison.json").read_text())
            decode = [
                item
                for item in result["comparisons"]
                if item["workload_family"] == "decode_heavy"
            ][0]
            self.assertEqual(decode["ratios"]["latency_ms"]["cc_on_over_cc_off"], 3.0)
            self.assertEqual(decode["ratio_of_ratios"]["latency_ms"], 1.5)


if __name__ == "__main__":
    unittest.main()
