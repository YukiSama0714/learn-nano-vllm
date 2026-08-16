# nano-vLLM v3：统一调度、细粒度 PagedAttention 与无损推测解码

本文对应 `slo_aware_v3`、`triton_paged` 和 greedy speculative decoding。
它同时是一份代码导读和 RTX 5090 验收手册。文中的性能阈值是待验证目标；
没有 5090 JSON 支撑的结果不会写成结论。

## 1. 先理解三个不变量

### 1.1 调度的单位是 request-token，不再是全局 phase

`ScheduledRequest` 为每个请求记录本步 token 数、prefill/decode 身份和是否
需要采样。`SchedulerOutput` 只负责汇总。旧策略仍通过 phase-exclusive
兼容层输出，因此原命令和原调度顺序不变。

v3 每步先按 TPOT slack 排序 running 请求，为其保留 decode 工作；剩余的
sequence 槽和 token budget 再交给按 normalized TTFT slack 排序的 prefill。
多个 prompt 可以在同一步各推进一个真实长度的 chunk。

模型输入被拼成一个 varlen batch：decode 的 query length 为 1，partial
prefill 的 query length 是本次 chunk 长度。只有 `logits_indices` 指向的位置
进入 LM Head，所以未完成的 partial prefill 不做大词表投影。

### 1.2 block table 是逻辑地址，KV tensor 是物理存储

BlockPool 管理逻辑 token 到物理 slot 的映射。refcount 表示共享，refcount
降到 0 的 block 进入 O(1) LRU 队列，但 committed full block 的 hash 仍可被
prefix cache 查找。同一个 hash 可以对应多个物理 block，查找时还会比较
token IDs，不能把 hash 相等直接当成内容相等。

`triton_paged` 对每个 query/token head：

1. 由 `query_to_request` 找到请求；
2. 由 `query_positions` 得到严格因果的 context length；
3. 经 block table 将 logical page 映射到 physical page；
4. 将 Q head 按 GQA group 映射到 KV head；
5. 用 FP32 online softmax 累积，最后转回 BF16。

当前支持 16、32、64-token page。FlashAttention 仍是默认后端和 reference。

### 1.3 speculative 的“提交长度”不等于“验证长度”

假设已有最后一个未计算 token，draft 提出 K 个 token。target 一次输入
`[last_token, draft_1, ..., draft_K]`，得到 K 个验证 logits 和一个 bonus
logit。

- 首次不一致：输出此前 accepted token 加 target replacement；
- 全部一致：输出 K 个 accepted token 加 bonus；
- target KV 只提交 `1 + accepted` 个位置；
- replacement/bonus 留作下一步的最后一个未计算 token；
- rejected lookahead 的完整 block 由 `truncate()` 释放，尾块允许覆盖；
- prefix hash 只覆盖提交后的完整 block。

因此 speculative 开关不改变 greedy token IDs，只改变一次 target forward
可能提交多少输出。`ngram` 从已有 token 中找最长 suffix；`draft` 使用独立的
Qwen3 draft 权重和 KV tensor，但与 target 共享逻辑 block 生命周期。显存
planner 在分配前把 target 与 draft 的每-block 字节数相加。

## 2. 代码入口

| 能力 | 入口 |
|---|---|
| 调度输出协议 | `nanovllm/engine/outputs.py` |
| v3 slack/budget/mixed batch | `nanovllm/engine/scheduler.py` |
| BlockPool、prefix、truncate | `nanovllm/engine/block_manager.py` |
| BatchMetadata、varlen packing | `nanovllm/engine/model_runner.py` |
| AttentionBackend、Triton kernel | `nanovllm/layers/attention.py` |
| n-gram 与 greedy verifier | `nanovllm/engine/spec_decode.py` |
| 指标与可复现 JSON | `benchmarks/benchmark_slo.py` |

## 3. 正确性测试

先运行普通单测：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

再在 5090 上运行 PagedAttention smoke test：

```bash
.venv/bin/python -m unittest -v tests.test_paged_attention
```

完整的 page/batch/context 矩阵需要显式开启，避免每次开发都分配大 tensor：

```bash
NANOVLLM_RUN_EXTENDED_KERNEL_TESTS=1 \
  .venv/bin/python -m unittest -v tests.test_paged_attention
```

覆盖范围为 page 16/32/64，batch 1/8/32/128，context
128/512/2048/4096，Qwen3 GQA 32Q/8KV、D=128，BF16 容差
`atol=rtol=2e-2`。

## 4. 冻结 greedy 金标

以下命令保存固定 100 条请求的 token IDs。模型路径按服务器实际位置修改。

```bash
RUN_ROOT=/nano-vllm/5090-runs/v3
MODEL_8B=/nano-vllm/models/Qwen3-8B
mkdir -p "$RUN_ROOT/golden"

.venv/bin/python benchmarks/benchmark_slo.py \
  --model "$MODEL_8B" \
  --num-requests 100 \
  --max-num-seqs 8 \
  --input-lens 128,512,2048 \
  --output-len 128 \
  --repeats 1 \
  --arrival-pattern poisson \
  --request-rate 2 \
  --scheduling-policy slo_aware_v2 \
  --temperature 0 \
  --output "$RUN_ROOT/golden/v2-flash.json"
```

每个结果的 `runs[].output_token_ids` 是金标；`environment` 同时保存 Git
commit、GPU、Torch/CUDA/Triton/FlashAttention 和模型结构摘要。

## 5. 阶段 1：v3 mixed batch A/B

先只改 scheduler，不改 attention 和 speculation：

```bash
for POLICY in slo_aware_v2 slo_aware_v3; do
  .venv/bin/python benchmarks/benchmark_slo.py \
    --model "$MODEL_8B" \
    --num-requests 100 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 4096 \
    --input-lens 128,512,2048 \
    --output-len 128 \
    --repeats 3 \
    --arrival-pattern poisson \
    --request-rate 2 \
    --scheduling-policy "$POLICY" \
    --prefill-chunk-size 1024 \
    --ttft-slo-ms 500 \
    --tpot-slo-ms 75 \
    --temperature 0 \
    --output "$RUN_ROOT/${POLICY}.json"
done
```

不要只看 mixed-step rate。验收同时要求 TTFT、Max ITL、E2E、output tok/s、
queue、preemption 和 starvation。Poisson 8/12/16 req/s 只改
`--request-rate`，其余参数保持不变。

## 6. 阶段 2：PagedAttention A/B

先跑 kernel microbenchmark：

```bash
.venv/bin/python benchmarks/kernels/benchmark_paged_attention.py \
  --page-sizes 16 32 64 \
  --batch-sizes 1 8 32 128 \
  --context-lengths 128 512 2048 4096 \
  --output "$RUN_ROOT/paged-attention-kernel.json"
```

端到端 A/B 只改 backend 和 block size。第一轮两边都加
`--enforce-eager`，隔离 kernel；第二轮再将 FlashAttention 的 CUDA Graph
作为生产基线比较。

```bash
# reference
--attention-backend flash_attn --kvcache-block-size 256 --enforce-eager

# candidate
--attention-backend triton_paged --kvcache-block-size 16 --enforce-eager
```

shared-prefix=240 的专项实验：

```bash
--input-len 512 --shared-prefix-len 240
```

block16 应产生 committed full-block cache hit；block256 的首块混入随机后缀，
命中率应为 0。与此同时比较 `kv_tail_waste_tokens`，不能只报告理论 16 倍。

## 7. 阶段 3：greedy speculative A/B

n-gram 不需要第二个模型：

```bash
--scheduling-policy slo_aware_v3 \
--temperature 0 \
--speculative-method ngram \
--num-speculative-tokens 4 \
--ngram-min 2 --ngram-max 5
```

Qwen3-0.6B draft：

```bash
MODEL_06B=/nano-vllm/models/Qwen3-0.6B

--scheduling-policy slo_aware_v3 \
--temperature 0 \
--speculative-method draft \
--draft-model "$MODEL_06B" \
--num-speculative-tokens 4
```

candidate 完成后逐 token 对比：

```bash
.venv/bin/python benchmarks/compare_greedy_outputs.py \
  "$RUN_ROOT/golden/v2-flash.json" \
  "$RUN_ROOT/draft.json"
```

只有输出完全一致后才讨论 acceptance 和 TPOT。高接受率 workload 要求 TPOT
改善至少 10%；普通随机 workload 变慢时保持 opt-in，并原样记录。

## 8. 汇总表与验收

```bash
.venv/bin/python benchmarks/compare_results.py \
  --eval-ttft-slo-ms 500 \
  --eval-itl-slo-ms 75 \
  "$RUN_ROOT/slo_aware_v2.json" \
  "$RUN_ROOT/slo_aware_v3.json" \
  "$RUN_ROOT/triton-paged.json" \
  "$RUN_ROOT/draft.json" \
  | tee "$RUN_ROOT/comparison.md"
```

最终报告要区分三类结论：正确性已通过、5090 性能已达到阈值、仍待调优。
本机 CPU 单测或 microbenchmark 不能代替端到端 5090 数据。
