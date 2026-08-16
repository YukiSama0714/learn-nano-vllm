import pickle
import unittest

from nanovllm import SamplingParams
from nanovllm.engine.outputs import ScheduledRequest, SchedulerOutput
from nanovllm.engine.sequence import Sequence


class SchedulerOutputTest(unittest.TestCase):
    def test_mixed_output_aggregates_per_request_tokens(self):
        prefill = Sequence([1, 2, 3])
        decode = Sequence([4, 5])
        output = SchedulerOutput(
            [
                ScheduledRequest(prefill, 3, True, True),
                ScheduledRequest(decode, 1, False, True),
            ]
        )

        self.assertTrue(output.is_mixed)
        self.assertEqual(output.num_prefill_tokens, 3)
        self.assertEqual(output.num_decode_tokens, 1)
        self.assertIsNone(output.legacy_is_prefill)

    def test_sequence_worker_state_preserves_greedy_temperature(self):
        sequence = Sequence(
            [1, 2],
            SamplingParams(temperature=0, max_tokens=4),
        )

        restored = pickle.loads(pickle.dumps(sequence))

        self.assertEqual(restored.temperature, 0)
        self.assertEqual(restored.token_ids, [1, 2])


if __name__ == "__main__":
    unittest.main()
