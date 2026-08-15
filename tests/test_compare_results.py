import unittest

from benchmarks.compare_results import request_violation_rate


class CompareResultsTest(unittest.TestCase):

    def test_request_violation_rate_uses_fixed_threshold(self):
        metrics = [
            {"max_inter_token_gap_ms": 50.0},
            {"max_inter_token_gap_ms": 80.0},
            {"max_inter_token_gap_ms": None},
        ]

        rate = request_violation_rate(
            metrics,
            "max_inter_token_gap_ms",
            target=75.0,
        )

        self.assertEqual(rate, 0.5)


if __name__ == "__main__":
    unittest.main()
