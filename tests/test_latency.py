import unittest

from gpu_cc_moe_inference.latency import categorize_module


class LatencyHookTests(unittest.TestCase):
    def test_categorize_qwen3_moe_modules(self):
        self.assertEqual(categorize_module("model.layers.0.self_attn"), "attention")
        self.assertEqual(categorize_module("model.layers.0.mlp.gate"), "router")
        self.assertEqual(categorize_module("model.layers.0.mlp"), "moe_ffn")
        self.assertIsNone(categorize_module("model.layers.0.mlp.experts.0.gate_proj"))


if __name__ == "__main__":
    unittest.main()
