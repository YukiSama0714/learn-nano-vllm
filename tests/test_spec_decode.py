import unittest

from nanovllm.engine.spec_decode import NgramProposer, verify_greedy_tokens


class NgramProposerTest(unittest.TestCase):
    def test_uses_longest_suffix_match(self):
        proposer = NgramProposer(min_size=2, max_size=5)

        result = proposer.propose([1, 2, 3, 1, 2], max_tokens=4)

        self.assertEqual(result, [3, 1, 2])

    def test_no_match_returns_empty_proposal(self):
        proposer = NgramProposer(min_size=2, max_size=5)

        self.assertEqual(proposer.propose([1, 2, 3, 4], 4), [])


class GreedyVerificationTest(unittest.TestCase):
    def test_all_accepted_adds_bonus_token(self):
        result = verify_greedy_tokens([2, 3], [2, 3, 4])

        self.assertEqual(result.output_token_ids, [2, 3, 4])
        self.assertEqual(result.accepted_tokens, 2)

    def test_first_rejection_uses_target_replacement(self):
        result = verify_greedy_tokens([2, 3], [9, 3, 4])

        self.assertEqual(result.output_token_ids, [9])
        self.assertEqual(result.accepted_tokens, 0)

    def test_partial_acceptance_stops_at_first_rejection(self):
        result = verify_greedy_tokens([2, 3, 4], [2, 8, 4, 5])

        self.assertEqual(result.output_token_ids, [2, 8])
        self.assertEqual(result.accepted_tokens, 1)

    def test_requires_bonus_logit(self):
        with self.assertRaisesRegex(ValueError, "bonus"):
            verify_greedy_tokens([2, 3], [2, 3])


if __name__ == "__main__":
    unittest.main()
