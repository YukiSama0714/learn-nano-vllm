import os
from dataclasses import dataclass

from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    draft_hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    scheduling_policy: str = "prefill_first"
    prefill_chunk_size: int = 0
    ttft_slo_ms: float = 500.0
    tpot_slo_ms: float = 50.0
    max_consecutive_decode_steps: int = 8
    scheduler_cost_ema_alpha: float = 0.2
    rms_norm_backend: str = "compiled"
    attention_backend: str = "flash_attn"
    speculative_method: str = "none"
    num_speculative_tokens: int = 4
    draft_model: str | None = None
    ngram_min: int = 2
    ngram_max: int = 5
    record_token_diagnostics: bool = False

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert 1 <= self.tensor_parallel_size <= 8
        if self.scheduling_policy not in {
            "prefill_first",
            "slo_aware",
            "slo_aware_v2",
            "slo_aware_v3",
        }:
            raise ValueError(f"unknown scheduling policy: {self.scheduling_policy}")
        if self.attention_backend not in {"flash_attn", "triton_paged"}:
            raise ValueError(f"unknown attention backend: {self.attention_backend}")
        if self.attention_backend == "flash_attn":
            if self.kvcache_block_size % 256:
                raise ValueError(
                    "flash_attn requires kvcache_block_size divisible by 256"
                )
        elif self.kvcache_block_size not in {16, 32, 64}:
            raise ValueError("triton_paged requires kvcache_block_size in {16, 32, 64}")
        if self.speculative_method not in {"none", "ngram", "draft"}:
            raise ValueError(f"unknown speculative method: {self.speculative_method}")
        if (
            self.speculative_method != "none"
            and self.scheduling_policy != "slo_aware_v3"
        ):
            raise ValueError(
                "speculative decoding requires scheduling_policy='slo_aware_v3'"
            )
        assert self.num_speculative_tokens > 0
        assert 1 <= self.ngram_min <= self.ngram_max
        if self.speculative_method == "draft":
            if self.tensor_parallel_size != 1:
                raise ValueError("draft speculation only supports TP=1")
            if self.draft_model is None or not os.path.isdir(self.draft_model):
                raise ValueError("draft_model must be an existing model directory")
            self.draft_hf_config = AutoConfig.from_pretrained(self.draft_model)
        if self.prefill_chunk_size == 0:
            self.prefill_chunk_size = self.max_num_batched_tokens
        assert 0 < self.prefill_chunk_size <= self.max_num_batched_tokens
        assert self.ttft_slo_ms >= 0
        assert self.tpot_slo_ms > 0
        assert self.max_consecutive_decode_steps > 0
        assert 0 < self.scheduler_cost_ema_alpha <= 1
        assert self.rms_norm_backend in {"eager", "compiled", "triton"}
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(
            self.max_model_len,
            self.hf_config.max_position_embeddings,
        )
        if self.draft_hf_config is not None:
            self.max_model_len = min(
                self.max_model_len,
                self.draft_hf_config.max_position_embeddings,
            )
