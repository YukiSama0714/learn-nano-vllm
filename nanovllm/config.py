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
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    scheduling_policy: str = "prefill_first"
    prefill_chunk_size: int = 0
    ttft_slo_ms: float = 500.0
    max_consecutive_decode_steps: int = 8

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.scheduling_policy in {"prefill_first", "slo_aware"}
        if self.prefill_chunk_size == 0:
            self.prefill_chunk_size = self.max_num_batched_tokens
        assert 0 < self.prefill_chunk_size <= self.max_num_batched_tokens
        assert self.ttft_slo_ms >= 0
        assert self.max_consecutive_decode_steps > 0
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(
            self.max_model_len,
            self.hf_config.max_position_embeddings,
        )
