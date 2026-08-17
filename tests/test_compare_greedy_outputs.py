import unittest

from benchmarks.compare_greedy_outputs import (
    compare_token_ids,
    first_mismatch_locations,
    format_token_diagnostic,
)


class CompareGreedyOutputsTest(unittest.TestCase):
    def test_identical_outputs_pass(self):
        outputs = [[[1, 2], [3, 4]]]

        self.assertEqual(compare_token_ids(outputs, outputs), [])

    def test_reports_first_different_token(self):
        mismatches = compare_token_ids(
            [[[1, 2, 3]]],
            [[[1, 9, 3]]],
        )

        self.assertEqual(
            mismatches,
            ["repeat 0, request 0, token 1: expected 2, actual 9"],
        )

    def test_reports_first_mismatch_location_for_diagnostics(self):
        locations = first_mismatch_locations(
            [[[1, 2, 3], [4, 5]]],
            [[[1, 9, 3], [4, 5]]],
        )

        self.assertEqual(locations, [(0, 0, 1)])

    def test_formats_logit_margin_context(self):
        formatted = format_token_diagnostic(
            "actual",
            {
                "top_token_ids": [9, 2],
                "top_logits": [4.5, 4.25],
                "margin": 0.25,
                "mixed_step": True,
                "is_prefill": False,
                "query_length": 1,
                "context_length": 12,
            },
        )

        self.assertIn("margin=0.25", formatted)
        self.assertIn("mixed=True", formatted)
        self.assertIn("q/context=1/12", formatted)


if __name__ == "__main__":
    unittest.main()
