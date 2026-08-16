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

SLO v1 策略（保留为负实验基线）：

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

SLO v2 策略：

```bash
.venv/bin/python benchmarks/benchmark_slo.py \
  --model /nano-vllm/models/Qwen3-8B \
  --num-requests 32 \
  --max-num-seqs 8 \
  --input-len 1024 \
  --output-len 128 \
  --repeats 5 \
  --scheduling-policy slo_aware_v2 \
  --prefill-chunk-size 1024 \
  --ttft-slo-ms 500 \
  --tpot-slo-ms 50 \
  --max-consecutive-decode-steps 8 \
  --output /nano-vllm/5090-runs/slo-aware-v2.json
```

### 已记录的 0.6B 负实验

RTX 5090、32 个请求同时到达、1024 输入、128 输出、重复 3 次时，v1 相比
`prefill_first` 得到：

| 指标 | prefill_first | slo_aware v1 |
|---|---:|---:|
| TTFT P95 ms | 607.02 | 1098.27 |
| TPOT P95 ms | 11.70 | 13.22 |
| E2E mean ms | 1494.47 | 2213.65 |
| Queue mean ms | 304.75 | 730.31 |
| Decode P95 ms | 446.05 | 328.09 |
| Prefill chunks P95 | 1 | 4 |
| Output tok/s | 2055.78 | 1768.20 |

v1 降低了 decode 执行时间的尾部波动，但固定 256-token 小 chunk 和延迟
请求准入使用户可见指标全面退化。这个结果是 v2 的设计依据，不能删除或只
报告优化后的数字。

重点比较：

| 指标 | 希望观察的问题 |
|---|---|
| TTFT P50/P95 | 新请求是否更快获得首 token |
| TPOT P50/P95 | 长 prefill 是否造成输出卡顿 |
| ITL P95/Max | TPOT 平均值是否掩盖单次长卡顿 |
| SLO violation rate | 有多少请求真正超过目标 |
| Output tok/s | 延迟改善牺牲了多少总吞吐 |
| Queue P95 | waiting queue 是否存在饥饿 |
| Prefill chunks | chunk 变小带来了多少额外调用 |
| Peak memory | 策略是否改变 KV cache 压力 |

## 3. 在线到达实验

bulk arrival 只代表离线批处理。正式评价 v2 时还要固定请求率，分别运行三种
策略。先用 0.6B 寻找低、中、高三个负载点，再在 8B 上复现。

示例：Poisson 8 req/s。

```bash
.venv/bin/python benchmarks/benchmark_slo.py \
  --model /nano-vllm/models/Qwen3-0.6B \
  --num-requests 32 \
  --max-num-seqs 8 \
  --input-len 1024 \
  --output-len 128 \
  --repeats 3 \
  --arrival-pattern poisson \
  --request-rate 8 \
  --scheduling-policy slo_aware_v2 \
  --prefill-chunk-size 1024 \
  --ttft-slo-ms 500 \
  --tpot-slo-ms 50 \
  --output /nano-vllm/5090-runs/online-v2.json
```

低负载时三种策略都应接近零排队；高负载时应同时报告吞吐和违反率，不能用
更低的实际完成请求率换取看似更好的延迟。

## 4. Chunk size 消融

保持其他参数不变，分别设置：

```text
prefill_chunk_size = 256, 512, 1024, 2048
```

不要只选择吞吐最高的配置。应根据项目目标选择“满足 TTFT/TPOT 目标时
吞吐最高”的配置。

## 5. Prefix cache 实验

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

## 6. KV-cache kernel 实验

```bash
.venv/bin/python benchmarks/kernels/benchmark_store_kvcache.py \
  --num-tokens 1 8 64 512 4096 \
  --num-kv-heads 8 \
  --head-dim 128 \
  --output /nano-vllm/5090-runs/store-kvcache.json
```

`1/8` 近似 decode 场景，`512/4096` 近似 prefill 场景。修改 kernel 后必须
同时满足：逐元素结果一致、各尺度无明显回退、端到端指标能够解释。

微基准完成后，用相同 workload 做端到端 A/B，两次命令只改：

```bash
--kv-store-backend triton
--kv-store-backend pytorch
```

PyTorch 后端是为 naive/optimized 对照保留的 benchmark-only 基线，
不支持 `--shared-prefix-len` 产生的 prefix-cache 命中，也不改变推理
引擎默认使用 Triton 的行为。

## 7. RMSNorm 算子实验

Q/K Norm 的 `D=128` 和主干 Norm 的 `D=4096` 分开测量：

```bash
.venv/bin/python benchmarks/kernels/benchmark_rmsnorm.py \
  --hidden-size 128 \
  --num-rows 32 256 4096 32768 \
  --rows-per-batch 32 \
  --batch-padding-rows 16 \
  --ops rms \
  --warmup 25 \
  --rep 100 \
  --output runs/kernels/rmsnorm-q-d128.json

.venv/bin/python benchmarks/kernels/benchmark_rmsnorm.py \
  --hidden-size 128 \
  --num-rows 8 64 512 4096 \
  --rows-per-batch 8 \
  --batch-padding-rows 40 \
  --ops rms \
  --warmup 25 \
  --rep 100 \
  --output runs/kernels/rmsnorm-k-d128.json

.venv/bin/python benchmarks/kernels/benchmark_rmsnorm.py \
  --hidden-size 4096 \
  --num-rows 1 8 64 512 4096 \
  --ops rms add_rms \
  --warmup 25 \
  --rep 100 \
  --output runs/kernels/rmsnorm-d4096.json
```

Qwen3-8B 的融合 QKV 投影每个 token 包含 32 个 Q heads、8 个 K
heads 和 8 个 V heads。前两条命令用 row padding 保留 `split`
产生的真实 batch stride，避免只测连续假数据。

每个 case 先与 eager 基线做正确性对比，再报告
`Triton / eager` 和 `Triton / torch.compile` 加速比。关键判断是后者：
原项目已经使用 `torch.compile`，因此只赢 eager 不构成替换理由。

## 8. 最终报告最少包含

1. 固定的软件版本、GPU、模型和 workload。
2. 三种策略的 P50/P95 TTFT、TPOT、ITL、吞吐、违反率和显存。
3. Chunk size 消融曲线或表格。
4. Prefix cache 的命中率与收益。
5. Kernel microbenchmark 与端到端结果的联系。
6. v1 负实验、根因分析，以及 v2 是否修复对应退化。
