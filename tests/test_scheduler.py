import unittest
from time import perf_counter
from types import SimpleNamespace

from nanovllm import SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


def make_config(
    policy="prefill_first",
    chunk_size=8,
    ttft_slo_ms=500.0,
    tpot_slo_ms=50.0,
    max_num_seqs=4,
    cost_ema_alpha=0.2,
    speculative_method="none",
    num_speculative_tokens=4,
):
    return SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=8,
        prefill_chunk_size=chunk_size,
        scheduling_policy=policy,
        ttft_slo_ms=ttft_slo_ms,
        tpot_slo_ms=tpot_slo_ms,
        max_consecutive_decode_steps=2,
        scheduler_cost_ema_alpha=cost_ema_alpha,
        eos=-1,
        kvcache_block_size=4,
        num_kvcache_blocks=32,
        speculative_method=speculative_method,
        num_speculative_tokens=num_speculative_tokens,
        ngram_min=2,
        ngram_max=5,
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

    def test_v2_admits_request_when_ttft_is_more_urgent(self):
        scheduler = Scheduler(
            make_config(
                policy="slo_aware_v2",
                chunk_size=8,
                max_num_seqs=2,
            )
        )
        self.add_running_sequence(scheduler)
        waiting = Sequence(list(range(12)), self.sampling_params)
        scheduler.add(waiting)
        scheduler.prefill_seconds_per_token = 0.001

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [waiting])
        self.assertEqual(waiting.num_scheduled_tokens, 8)

    def test_v2_protects_overdue_decode(self):
        scheduler = Scheduler(
            make_config(
                policy="slo_aware_v2",
                ttft_slo_ms=1_000,
                tpot_slo_ms=10,
                max_num_seqs=1,
            )
        )
        running = self.add_running_sequence(scheduler)
        running.metrics.mark_token(1, perf_counter() - 0.1)
        waiting = Sequence(list(range(8)), self.sampling_params)
        scheduler.add(waiting)

        scheduled, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [running])

    def test_v2_prioritizes_overdue_ttft(self):
        scheduler = Scheduler(
            make_config(
                policy="slo_aware_v2",
                ttft_slo_ms=10,
                tpot_slo_ms=1_000,
                max_num_seqs=1,
            )
        )
        running = self.add_running_sequence(scheduler)
        running.metrics.mark_token(1, perf_counter())
        waiting = Sequence(
            list(range(8)),
            self.sampling_params,
            arrival_time=perf_counter() - 0.1,
        )
        scheduler.add(waiting)

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [waiting])

    def test_v2_sizes_chunk_from_decode_slack(self):
        scheduler = Scheduler(
            make_config(
                policy="slo_aware_v2",
                chunk_size=8,
                ttft_slo_ms=1_000,
                tpot_slo_ms=10,
                max_num_seqs=1,
            )
        )
        running = self.add_running_sequence(scheduler)
        running.metrics.mark_token(1, perf_counter())
        scheduler.add(Sequence(list(range(12)), self.sampling_params))
        scheduler.prefill_seconds_per_token = 0.002

        self.assertEqual(scheduler._prefill_token_budget_v2(), 4)

    def test_v2_updates_cost_estimates_with_ema(self):
        scheduler = Scheduler(make_config(policy="slo_aware_v2", cost_ema_alpha=0.5))
        seq = Sequence(list(range(8)), self.sampling_params)
        seq.num_scheduled_tokens = 8

        scheduler._update_cost_estimates([seq], True, 0.08)
        scheduler._update_cost_estimates([seq], True, 0.16)
        scheduler._update_cost_estimates([seq], False, 0.04)

        self.assertAlmostEqual(scheduler.prefill_seconds_per_token, 0.015)
        self.assertAlmostEqual(scheduler.decode_step_seconds, 0.04)

        scheduler.reset_cost_estimates()

        self.assertIsNone(scheduler.prefill_seconds_per_token)
        self.assertIsNone(scheduler.decode_step_seconds)

    def test_v2_prefills_least_lax_waiting_request_first(self):
        scheduler = Scheduler(make_config(policy="slo_aware_v2", chunk_size=8))
        arrival_time = perf_counter()
        short = Sequence(
            list(range(4)),
            self.sampling_params,
            arrival_time=arrival_time,
        )
        long = Sequence(
            list(range(12)),
            self.sampling_params,
            arrival_time=arrival_time,
        )
        scheduler.add(short)
        scheduler.add(long)
        scheduler.prefill_seconds_per_token = 0.001

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [long])

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

    def test_v3_mixes_decode_and_prefill_in_one_step(self):
        scheduler = Scheduler(
            make_config(
                policy="slo_aware_v3",
                chunk_size=4,
                max_num_seqs=2,
            )
        )
        running = self.add_running_sequence(scheduler)
        waiting = Sequence(list(range(12)), self.sampling_params)
        scheduler.add(waiting)

        output = scheduler.schedule()

        self.assertTrue(output.is_mixed)
        self.assertEqual(output.num_decode_tokens, 1)
        self.assertEqual(output.num_prefill_tokens, 4)
        self.assertEqual(
            {request.sequence for request in output.scheduled_requests},
            {running, waiting},
        )

    def test_v3_reserves_decode_before_more_urgent_prefill(self):
        scheduler = Scheduler(
            make_config(
                policy="slo_aware_v3",
                chunk_size=8,
                max_num_seqs=1,
                ttft_slo_ms=1,
            )
        )
        running = self.add_running_sequence(scheduler)
        waiting = Sequence(
            list(range(12)),
            self.sampling_params,
            arrival_time=perf_counter() - 1,
        )
        scheduler.add(waiting)

        output = scheduler.schedule()

        self.assertTrue(output.is_decode_only)
        self.assertEqual(output.sequences, [running])

    def test_v3_sizes_prefill_against_selected_decode_budget(self):
        scheduler = Scheduler(
            make_config(
                policy="slo_aware_v3",
                chunk_size=8,
                max_num_seqs=2,
                tpot_slo_ms=10,
            )
        )
        running = self.add_running_sequence(scheduler)
        running.metrics.mark_token(1, perf_counter())
        scheduler.add(Sequence(list(range(12)), self.sampling_params))
        scheduler.prefill_seconds_per_token = 0.002
        scheduler.decode_step_seconds = 0.002

        output = scheduler.schedule()
        prefill_request = next(
            request for request in output.scheduled_requests if request.is_prefill
        )

        self.assertEqual(prefill_request.num_scheduled_tokens, 4)

    def test_v3_allows_multiple_partial_prefills(self):
        scheduler = Scheduler(
            make_config(
                policy="slo_aware_v3",
                chunk_size=4,
                max_num_seqs=2,
            )
        )
        first = Sequence(list(range(12)), self.sampling_params)
        second = Sequence(list(range(12, 24)), self.sampling_params)
        scheduler.add(first)
        scheduler.add(second)

        output = scheduler.schedule()

        self.assertTrue(output.is_prefill_only)
        self.assertEqual(len(output.scheduled_requests), 2)
        self.assertEqual(output.num_prefill_tokens, 8)
        self.assertFalse(
            any(request.needs_sampling for request in output.scheduled_requests)
        )
        scheduler.postprocess(output, [None, None])
        self.assertEqual(first.num_cached_tokens, 4)
        self.assertEqual(second.num_cached_tokens, 4)

    def test_v3_records_mixed_cost_bucket(self):
        scheduler = Scheduler(
            make_config(
                policy="slo_aware_v3",
                chunk_size=4,
                max_num_seqs=2,
            )
        )
        self.add_running_sequence(scheduler)
        scheduler.add(Sequence(list(range(12)), self.sampling_params))
        output = scheduler.schedule()

        scheduler.postprocess(output, [7, None], step_duration=0.02)

        key = (
            scheduler._cost_bucket(output.num_prefill_tokens),
            scheduler._cost_bucket(1),
        )
        self.assertEqual(scheduler.mixed_step_seconds[key], 0.02)

    def test_v3_ngram_partial_acceptance_commits_only_verified_kv(self):
        scheduler = Scheduler(
            make_config(
                policy="slo_aware_v3",
                max_num_seqs=1,
                speculative_method="ngram",
            )
        )
        sequence = Sequence(
            [1, 2, 3, 1, 2],
            SamplingParams(
                temperature=0,
                max_tokens=8,
                ignore_eos=True,
            ),
        )
        scheduler.block_manager.allocate(sequence, num_cached_blocks=0)
        sequence.num_cached_tokens = len(sequence) - 1
        sequence.status = SequenceStatus.RUNNING
        scheduler.running.append(sequence)

        output = scheduler.schedule()

        request = output.scheduled_requests[0]
        self.assertEqual(request.speculative_token_ids, [3, 1, 2])
        scheduler.postprocess(output, [[3, 9, 8, 7]])
        self.assertEqual(sequence.completion_token_ids, [3, 9])
        self.assertEqual(sequence.num_cached_tokens, 6)
        self.assertEqual(request.accepted_tokens, 1)
        self.assertEqual(sequence.metrics.proposed_tokens, 3)
        self.assertEqual(sequence.metrics.accepted_tokens, 1)

    def test_speculation_rejects_non_greedy_request(self):
        scheduler = Scheduler(
            make_config(
                policy="slo_aware_v3",
                speculative_method="ngram",
            )
        )

        with self.assertRaisesRegex(ValueError, "greedy"):
            scheduler.add(Sequence([1, 2, 1, 2], self.sampling_params))

    def test_v3_draft_reserves_verification_tokens(self):
        scheduler = Scheduler(
            make_config(
                policy="slo_aware_v3",
                max_num_seqs=1,
                speculative_method="draft",
                num_speculative_tokens=4,
            )
        )
        sequence = Sequence(
            [1, 2, 3, 4],
            SamplingParams(
                temperature=0,
                max_tokens=8,
                ignore_eos=True,
            ),
        )
        scheduler.block_manager.allocate(sequence, num_cached_blocks=0)
        sequence.num_cached_tokens = len(sequence) - 1
        sequence.status = SequenceStatus.RUNNING
        scheduler.running.append(sequence)

        output = scheduler.schedule()
        request = output.scheduled_requests[0]

        self.assertEqual(request.speculative_method, "draft")
        self.assertEqual(request.num_scheduled_tokens, 5)
        self.assertEqual(len(sequence.block_table), 2)

    def test_speculative_eos_stops_accepted_suffix(self):
        scheduler = Scheduler(
            make_config(
                policy="slo_aware_v3",
                max_num_seqs=1,
                speculative_method="ngram",
            )
        )
        scheduler.eos = 3
        sequence = Sequence(
            [1, 2, 3, 1, 2],
            SamplingParams(temperature=0, max_tokens=8),
        )
        scheduler.block_manager.allocate(sequence, num_cached_blocks=0)
        sequence.num_cached_tokens = len(sequence) - 1
        sequence.status = SequenceStatus.RUNNING
        scheduler.running.append(sequence)
        output = scheduler.schedule()

        scheduler.postprocess(output, [[3, 1, 2, 8]])

        self.assertTrue(sequence.is_finished)
        self.assertEqual(sequence.completion_token_ids, [3])
        self.assertEqual(len(sequence.block_table), 0)

    def test_speculation_respects_max_tokens(self):
        scheduler = Scheduler(
            make_config(
                policy="slo_aware_v3",
                max_num_seqs=1,
                speculative_method="ngram",
            )
        )
        sequence = Sequence(
            [1, 2, 3, 1, 2],
            SamplingParams(
                temperature=0,
                max_tokens=2,
                ignore_eos=True,
            ),
        )
        scheduler.block_manager.allocate(sequence, num_cached_blocks=0)
        sequence.num_cached_tokens = len(sequence) - 1
        sequence.status = SequenceStatus.RUNNING
        scheduler.running.append(sequence)
        output = scheduler.schedule()

        scheduler.postprocess(output, [[3, 9]])

        self.assertTrue(sequence.is_finished)
        self.assertEqual(sequence.completion_token_ids, [3, 9])


if __name__ == "__main__":
    unittest.main()
