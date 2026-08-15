import unittest
from types import SimpleNamespace

from nanovllm import SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


def make_config(
    policy="prefill_first",
    chunk_size=8,
    ttft_slo_ms=500.0,
    max_num_seqs=4,
):
    return SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=8,
        prefill_chunk_size=chunk_size,
        scheduling_policy=policy,
        ttft_slo_ms=ttft_slo_ms,
        max_consecutive_decode_steps=2,
        eos=-1,
        kvcache_block_size=4,
        num_kvcache_blocks=32,
    )


class SchedulerTest(unittest.TestCase):

    def setUp(self):
        self.original_block_size = Sequence.block_size
        Sequence.block_size = 4
        self.sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=4,
            ignore_eos=True,
        )

    def tearDown(self):
        Sequence.block_size = self.original_block_size

    def add_running_sequence(self, scheduler):
        seq = Sequence([1, 2], self.sampling_params)
        scheduler.block_manager.allocate(seq, num_cached_blocks=0)
        seq.status = SequenceStatus.RUNNING
        scheduler.running.append(seq)
        return seq

    def test_chunk_size_bounds_prefill_work(self):
        scheduler = Scheduler(make_config(chunk_size=4))
        seq = Sequence(list(range(12)), self.sampling_params)
        scheduler.add(seq)

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [seq])
        self.assertEqual(seq.num_scheduled_tokens, 4)
        self.assertEqual(seq.metrics.prefill_chunks, 1)

    def test_slo_policy_interleaves_prefill_and_decode(self):
        scheduler = Scheduler(
            make_config(
                policy="slo_aware",
                chunk_size=4,
                ttft_slo_ms=0,
            )
        )
        running = self.add_running_sequence(scheduler)
        waiting = Sequence(list(range(12)), self.sampling_params)
        scheduler.add(waiting)

        scheduled, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [waiting])
        scheduler.postprocess(
            scheduled,
            token_ids=[10],
            is_prefill=True,
            step_finished_at=waiting.metrics.arrival_time + 0.1,
        )

        scheduled, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [running])

    def test_slo_policy_uses_small_chunk_before_ttft_deadline(self):
        scheduler = Scheduler(
            make_config(
                policy="slo_aware",
                chunk_size=8,
                ttft_slo_ms=100_000,
            )
        )
        self.add_running_sequence(scheduler)
        waiting = Sequence(list(range(12)), self.sampling_params)
        scheduler.add(waiting)
        scheduler.consecutive_decode_steps = 2

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [waiting])
        self.assertEqual(waiting.num_scheduled_tokens, 4)

    def test_slo_policy_rotates_decode_sequences(self):
        scheduler = Scheduler(
            make_config(
                policy="slo_aware",
                max_num_seqs=2,
            )
        )
        first = self.add_running_sequence(scheduler)
        second = self.add_running_sequence(scheduler)
        third = self.add_running_sequence(scheduler)

        scheduled, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [first, second])
        self.assertEqual(list(scheduler.running), [third, first, second])

    def test_prefix_cache_hit_is_reported(self):
        scheduler = Scheduler(make_config())
        first = Sequence(
            list(range(8)),
            SamplingParams(
                temperature=1.0,
                max_tokens=1,
                ignore_eos=True,
            ),
        )
        scheduler.add(first)
        scheduled, is_prefill = scheduler.schedule()
        scheduler.postprocess(scheduled, [10], is_prefill)

        second = Sequence(list(range(8)), self.sampling_params)
        scheduler.add(second)
        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [second])
        self.assertEqual(second.metrics.prefix_cache_hit_tokens, 4)


if __name__ == "__main__":
    unittest.main()
