# RTX 5090 四天实验计划

目标不是跑出一个最大的 tokens/s，而是得到可以复现、可以解释的
吞吐—延迟—显存权衡。

## 1. 正确性检查

所有命令都在项目根目录执行：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

先用 0.6B 做一次生成，确认每个输出新增了 `metrics` 字段；再使用 8B 跑
正式实验。

## 2. 调度策略 A/B 实验

固定工作负载：32 个请求、1024 输入 token、128 输出 token、最多 8 个
sequence 同批 decode。每组重复 5 次。

原始策略：

```bash
.venv/bin/python benchmarks/benchmark_slo.py \
  --model /nano-vllm/models/Qwen3-8B \
  --num-requests 32 \
  --max-num-seqs 8 \
  --input-len 1024 \
  --output-len 128 \
  --repeats 5 \
  --scheduling-policy prefill_first \
  --output /nano-vllm/5090-runs/prefill-first.json
```

SLO 策略：

```bash
.venv/bin/python benchmarks/benchmark_slo.py \
  --model /nano-vllm/models/Qwen3-8B \
  --num-requests 32 \
  --max-num-seqs 8 \
  --input-len 1024 \
  --output-len 128 \
  --repeats 5 \
  --scheduling-policy slo_aware \
  --prefill-chunk-size 1024 \
  --ttft-slo-ms 500 \
  --max-consecutive-decode-steps 8 \
  --output /nano-vllm/5090-runs/slo-aware.json
```

重点比较：

| 指标 | 希望观察的问题 |
|---|---|
| TTFT P50/P95 | 新请求是否更快获得首 token |
| TPOT P50/P95 | 长 prefill 是否造成输出卡顿 |
| Output tok/s | 延迟改善牺牲了多少总吞吐 |
| Queue P95 | waiting queue 是否存在饥饿 |
| Prefill chunks | chunk 变小带来了多少额外调用 |
| Peak memory | 策略是否改变 KV cache 压力 |

## 3. Chunk size 消融

保持其他参数不变，分别设置：

```text
prefill_chunk_size = 256, 512, 1024, 2048
```

不要只选择吞吐最高的配置。应根据项目目标选择“满足 TTFT/TPOT 目标时
吞吐最高”的配置。

## 4. Prefix cache 实验

随机 prompt 对照组使用 `--shared-prefix-len 0`。共享前缀实验使用：

```bash
.venv/bin/python benchmarks/benchmark_slo.py \
  --model /nano-vllm/models/Qwen3-8B \
  --num-requests 32 \
  --max-num-seqs 8 \
  --input-len 1024 \
  --output-len 128 \
  --shared-prefix-len 512 \
  --repeats 5 \
  --output /nano-vllm/5090-runs/prefix-512.json
```

检查 `prefix_cache_hit_rate`，再观察 TTFT 和 prefill 时间是否同步下降。

多组 JSON 可以直接整理成 Markdown 表格：

```bash
.venv/bin/python benchmarks/compare_results.py \
  /nano-vllm/5090-runs/prefill-first.json \
  /nano-vllm/5090-runs/slo-aware.json \
  /nano-vllm/5090-runs/prefix-512.json
```

## 5. KV-cache kernel 实验

```bash
.venv/bin/python benchmarks/kernels/benchmark_store_kvcache.py \
  --num-tokens 1 8 64 512 4096 \
  --num-kv-heads 8 \
  --head-dim 128 \
  --output /nano-vllm/5090-runs/store-kvcache.json
```

`1/8` 近似 decode 场景，`512/4096` 近似 prefill 场景。修改 kernel 后必须
同时满足：逐元素结果一致、各尺度无明显回退、端到端指标能够解释。

## 6. 最终报告最少包含

1. 固定的软件版本、GPU、模型和 workload。
2. 原策略与新策略的 P50/P95 TTFT、TPOT、吞吐和显存。
3. Chunk size 消融曲线或表格。
4. Prefix cache 的命中率与收益。
5. Kernel microbenchmark 与端到端结果的联系。
6. 一个失败或反直觉实验，以及原因分析。
