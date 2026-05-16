import unittest

from gpu_cc_moe_inference.cc_path import (
    gemm_flops,
    moe_io_roofline_rows,
    parse_size_bytes,
    stage_estimates,
    theoretical_ms_from_peak,
    transfer_roofline_rows,
)
from gpu_cc_moe_inference.schema import validate_cc_path_row


class CcPathTests(unittest.TestCase):
    def test_parse_size_bytes(self):
        self.assertEqual(parse_size_bytes("4KB"), 4000)
        self.assertEqual(parse_size_bytes("4KiB"), 4096)
        self.assertEqual(parse_size_bytes("1MB"), 1_000_000)

    def test_stage_estimates_validate(self):
        row = {
            "schema_version": "cc-path-v1",
            "run_id": "run",
            "direction": "H2D",
            "bytes": 4096,
            "cpu_wall_ms": 0.3,
            "cuda_event_ms": 0.1,
            "sync_wait_ms": 0.2,
            "driver_call_ms": 0.05,
            "driver_syscall_cpu_ms": 0.01,
        }
        row.update(
            stage_estimates(
                bytes_count=4096,
                cpu_wall_ms=0.3,
                cuda_event_ms=0.1,
                driver_call_ms=0.05,
                effective_bandwidth_gbps=4.0,
            )
        )
        validate_cc_path_row(row)
        self.assertTrue(row["estimate_provenance"]["gpu_decrypt_residual_est_ms"]["estimated"])

    def test_roofline_math(self):
        self.assertEqual(gemm_flops(2, 3, 4), 48)
        self.assertAlmostEqual(theoretical_ms_from_peak(1_000_000_000_000, 100.0), 10.0)
        rows = transfer_roofline_rows(sizes=[4_000_000], effective_bandwidth_gbps=4.0)
        self.assertAlmostEqual(rows[0]["theoretical_ms"], 1.0)
        moe_rows = moe_io_roofline_rows(
            token_counts=[2], effective_bandwidth_gbps=4.0, hidden_size=2048
        )
        self.assertTrue(any(row["operator"] == "hidden_states_h2d" for row in moe_rows))
        self.assertTrue(any(row["kind"] == "io_moe_shape" for row in moe_rows))


if __name__ == "__main__":
    unittest.main()
