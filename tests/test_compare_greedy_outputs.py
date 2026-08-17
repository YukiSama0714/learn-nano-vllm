import unittest

from benchmarks.compare_greedy_outputs import compare_token_ids


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


if __name__ == "__main__":
    unittest.main()
