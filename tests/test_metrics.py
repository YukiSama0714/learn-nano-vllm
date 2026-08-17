import unittest

from nanovllm.engine.metrics import RequestMetrics


class RequestMetricsTest(unittest.TestCase):
    def test_request_lifecycle_metrics(self):
        metrics = RequestMetrics(arrival_time=10.0)
        metrics.mark_scheduled(is_prefill=True, now=11.0)
        metrics.record_model_step(is_prefill=True, duration=0.4)
        metrics.record_prefix_cache_hit(128)
        metrics.mark_token(num_completion_tokens=1, now=12.0)
        metrics.mark_scheduled(is_prefill=False, now=12.5)
        metrics.record_model_step(is_prefill=False, duration=0.6)
        metrics.mark_token(num_completion_tokens=2, now=13.0)
        metrics.mark_finished(now=13.0)

        result = metrics.to_dict(
            num_prompt_tokens=256,
            num_completion_tokens=2,
        )

        self.assertEqual(result["ttft_ms"], 2000.0)
        self.assertEqual(result["tpot_ms"], 1000.0)
        self.assertEqual(result["e2e_ms"], 3000.0)
        self.assertEqual(result["queue_ms"], 1000.0)
        self.assertEqual(result["prefill_ms"], 400.0)
        self.assertEqual(result["decode_ms"], 600.0)
        self.assertEqual(result["inter_token_gap_p95_ms"], 1000.0)
        self.assertEqual(result["max_inter_token_gap_ms"], 1000.0)
        self.assertEqual(result["prefix_cache_hit_rate"], 0.5)
        self.assertEqual(result["prefill_chunks"], 1)
        self.assertEqual(result["decode_steps"], 1)

    def test_preemption_adds_queue_time(self):
        metrics = RequestMetrics(arrival_time=1.0)
        metrics.mark_scheduled(is_prefill=True, now=2.0)
        metrics.mark_preempted(now=3.0)

        self.assertEqual(metrics.current_queue_time(now=5.0), 3.0)

        metrics.mark_scheduled(is_prefill=True, now=5.0)
        result = metrics.to_dict(10, 0)

        self.assertEqual(result["queue_ms"], 3000.0)
        self.assertEqual(result["preemptions"], 1)

    def test_inter_token_metrics_keep_tail_stalls_visible(self):
        metrics = RequestMetrics(arrival_time=0.0)
        metrics.mark_token(1, now=1.0)
        metrics.mark_token(2, now=2.0)
        metrics.mark_token(3, now=5.0)
        metrics.mark_finished(now=5.0)

        result = metrics.to_dict(
            num_prompt_tokens=4,
            num_completion_tokens=3,
        )

        self.assertEqual(result["tpot_ms"], 2000.0)
        self.assertEqual(result["inter_token_gap_p95_ms"], 2900.0)
        self.assertEqual(result["max_inter_token_gap_ms"], 3000.0)

    def test_token_diagnostics_are_preserved_for_benchmark_output(self):
        metrics = RequestMetrics(arrival_time=0.0)
        diagnostic = {
            "top_token_ids": [4, 7],
            "top_logits": [2.0, 1.5],
            "margin": 0.5,
        }

        metrics.record_token_diagnostics([diagnostic])
        result = metrics.to_dict(4, 1)

        self.assertEqual(result["token_diagnostics"], [diagnostic])


if __name__ == "__main__":
    unittest.main()
