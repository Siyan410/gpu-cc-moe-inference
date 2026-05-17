import unittest

from gpu_cc_moe_inference.sidechannel import evaluate_sidechannel


def make_rows():
    labels = []
    features = []
    for prompt_id in range(6):
        for token_index in range(3):
            expert = (prompt_id + token_index) % 4
            labels.append(
                {
                    "schema_version": "routing-label-v1",
                    "run_id": "labels",
                    "prompt_id": str(prompt_id),
                    "layer": 0,
                    "token_index": token_index,
                    "topk": [expert, (expert + 1) % 4],
                }
            )
            features.append(
                {
                    "schema_version": "attacker-feature-v1",
                    "run_id": "features",
                    "prompt_id": str(prompt_id),
                    "token_index": token_index,
                    "features": {
                        "prompt_chars": float(10 + prompt_id),
                        "token_index": float(token_index),
                        "signal": float(expert),
                    },
                }
            )
    return labels, features


class SidechannelTests(unittest.TestCase):
    def test_evaluate_sidechannel_outputs_controls(self):
        labels, features = make_rows()
        result = evaluate_sidechannel(labels, features, seed=1, test_fraction=0.34, neighbors=3)
        self.assertEqual(result["schema_version"], "sidechannel-eval-v1")
        self.assertGreater(result["rows_joined"], 0)
        self.assertIn("topk_jaccard", result["metrics"])
        self.assertIn("per_expert_f1", result["metrics"])
        self.assertIn("label_shuffle", result["negative_controls"])
        self.assertIn("length_matched_prompt", result["negative_controls"])
        self.assertIn("main_minus_length_matched_prompt", result["negative_control_deltas"])


if __name__ == "__main__":
    unittest.main()
