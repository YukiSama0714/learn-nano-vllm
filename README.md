<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.
* 📊 **Request observability** - TTFT, TPOT, queue, prefill, decode, cache-hit, and preemption metrics
* 🎯 **SLO-aware scheduling** - Adaptive chunked prefill that balances new-request TTFT and active-request TPOT

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

Each output also contains request-level metrics:

```python
outputs[0]["metrics"]
# {
#   "ttft_ms": ...,
#   "tpot_ms": ...,
#   "e2e_ms": ...,
#   "queue_ms": ...,
#   "prefix_cache_hit_rate": ...,
#   ...
# }
```

The original scheduling behavior remains the default. Enable the experimental
policy explicitly:

```python
llm = LLM(
    "/YOUR/MODEL/PATH",
    scheduling_policy="slo_aware",
    prefill_chunk_size=1024,
    ttft_slo_ms=500,
    max_consecutive_decode_steps=8,
)
```

## Benchmark

See `bench.py` for the original throughput benchmark.

For repeatable request-level SLO experiments, see
[`benchmarks/benchmark_slo.py`](benchmarks/benchmark_slo.py). The benchmark
supports fixed workloads, repeated runs, shared-prefix workloads, JSON output,
and direct comparison between `prefill_first` and `slo_aware`.

The 5090 experiment commands and the Chinese source-reading guide are in:

* [`docs/rtx5090-experiments.md`](docs/rtx5090-experiments.md)
* [`docs/learning-guide-zh.md`](docs/learning-guide-zh.md)

**Upstream Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
