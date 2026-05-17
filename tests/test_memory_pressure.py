import unittest

from gpu_cc_moe_inference.memory_pressure import parse_float_list


class MemoryPressureTests(unittest.TestCase):
    def test_parse_float_list(self):
        self.assertEqual(parse_float_list("0,1.5,2"), [0.0, 1.5, 2.0])

    def test_parse_float_list_rejects_negative(self):
        with self.assertRaises(Exception):
            parse_float_list("-1")


if __name__ == "__main__":
    unittest.main()
