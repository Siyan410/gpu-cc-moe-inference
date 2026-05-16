import unittest

from gpu_cc_moe_inference.schema import (
    validate_attacker_feature_row,
    validate_routing_label_row,
)


class SchemaTests(unittest.TestCase):
    def test_label_schema(self):
        validate_routing_label_row(
            {
                "schema_version": "routing-label-v1",
                "run_id": "r",
                "prompt_id": "0",
                "layer": 1,
                "token_index": 2,
                "topk": [1, 2, 3],
            }
        )

    def test_feature_schema(self):
        validate_attacker_feature_row(
            {
                "schema_version": "attacker-feature-v1",
                "run_id": "r",
                "prompt_id": "0",
                "token_index": 2,
                "features": {"cpu_wall_ms": 1.0},
            }
        )

    def test_feature_schema_rejects_non_object(self):
        with self.assertRaises(ValueError):
            validate_attacker_feature_row(
                {
                    "schema_version": "attacker-feature-v1",
                    "run_id": "r",
                    "prompt_id": "0",
                    "token_index": 2,
                    "features": [],
                }
            )


if __name__ == "__main__":
    unittest.main()
