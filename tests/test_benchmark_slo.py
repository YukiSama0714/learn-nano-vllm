import random
import unittest

from benchmarks.benchmark_slo import (
    make_arrival_offsets,
    run_workload,
    violation_rate,
)


class FakeClock:

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, duration):
        self.now += duration


class FakeTokenizer:

    def decode(self, token_ids):
        return " ".join(str(token_id) for token_id in token_ids)


class FakeLlm:

    def __init__(self, clock):
        self.clock = clock
        self.tokenizer = FakeTokenizer()
        self.active = []
        self.arrival_times = []
        self.next_seq_id = 0

    def add_request(self, prompt, sampling_params, arrival_time):
        seq_id = self.next_seq_id
        self.next_seq_id += 1
        self.active.append(seq_id)
        self.arrival_times.append(arrival_time)
        return seq_id

    def is_finished(self):
        return not self.active

    def step(self):
        self.clock.advance(0.01)
        seq_id = self.active.pop(0)
        return [(seq_id, [seq_id])], -1

    def take_finished_request_metrics(self, seq_id):
        return {"seq_id": seq_id}


class BenchmarkSloTest(unittest.TestCase):

    def test_bulk_arrivals_share_the_same_offset(self):
        offsets = make_arrival_offsets(
            "bulk",
            request_rate=0,
            num_requests=4,
            rng=random.Random(1),
        )

        self.assertEqual(offsets, [0.0, 0.0, 0.0, 0.0])

    def test_constant_arrivals_follow_requested_rate(self):
        offsets = make_arrival_offsets(
            "constant",
            request_rate=2,
            num_requests=4,
            rng=random.Random(1),
        )

        self.assertEqual(offsets, [0.0, 0.5, 1.0, 1.5])

    def test_poisson_arrivals_are_seeded_and_monotonic(self):
        first = make_arrival_offsets(
            "poisson",
            request_rate=2,
            num_requests=4,
            rng=random.Random(1),
        )
        second = make_arrival_offsets(
            "poisson",
            request_rate=2,
            num_requests=4,
            rng=random.Random(1),
        )

        self.assertEqual(first, second)
        self.assertEqual(first[0], 0.0)
        self.assertEqual(first, sorted(first))

    def test_violation_rate_ignores_missing_values(self):
        rate = violation_rate([10.0, None, 30.0], target=20.0)

        self.assertEqual(rate, 0.5)

    def test_online_driver_injects_requests_at_scheduled_times(self):
        clock = FakeClock()
        llm = FakeLlm(clock)

        outputs, elapsed = run_workload(
            llm,
            prompts=[[1], [2], [3]],
            sampling_params=object(),
            arrival_offsets=[0.0, 0.5, 1.0],
            clock=clock,
            sleeper=clock.advance,
        )

        self.assertEqual(llm.arrival_times, [0.0, 0.5, 1.0])
        self.assertEqual(len(outputs), 3)
        self.assertAlmostEqual(elapsed, 1.01)

    def test_online_driver_rejects_unsorted_arrivals(self):
        clock = FakeClock()

        with self.assertRaisesRegex(ValueError, "monotonic"):
            run_workload(
                FakeLlm(clock),
                prompts=[[1], [2]],
                sampling_params=object(),
                arrival_offsets=[1.0, 0.0],
                clock=clock,
                sleeper=clock.advance,
            )


if __name__ == "__main__":
    unittest.main()
