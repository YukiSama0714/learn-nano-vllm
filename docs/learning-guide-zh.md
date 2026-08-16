# nano-vLLM SLO 项目学习导读

这份导读不要求先会 CUDA。建议按“请求生命周期 → 调度 → KV cache →
kernel”的顺序阅读，因为底层优化只有放回请求生命周期里才有意义。

## 1. 一次请求经过哪里

```text
LLMEngine.generate
  → add_request
  → Scheduler.schedule
  → ModelRunner.run
  → Qwen3ForCausalLM
  → Attention.forward
  → Scheduler.postprocess
```

对应文件：

1. `nanovllm/engine/llm_engine.py`：接收请求并驱动执行循环。
2. `nanovllm/engine/sequence.py`：保存一个请求的 token、状态和指标。
3. `nanovllm/engine/scheduler.py`：决定下一步执行 prefill 还是 decode。
4. `nanovllm/engine/model_runner.py`：准备张量、执行模型和 CUDA Graph。
5. `nanovllm/layers/attention.py`：FlashAttention 与 Triton KV 写入。
6. `nanovllm/engine/block_manager.py`：Paged KV cache 和 prefix cache。

## 2. Prefill 和 decode 为什么要分开

Prefill 一次处理整段输入，矩阵较大，GPU 并行度高，但长 prompt 会让正在
decode 的请求停顿。Decode 每个请求每轮只处理一个 token，单次计算较小，
需要依靠 continuous batching 提高 GPU 利用率。

```text
更多/更大的 prefill chunk
  → 新请求更快拿到首 token
  → 活跃请求的 token 间隔可能变大

更多连续 decode step
  → 活跃请求输出更流畅
  → 新请求可能长期排队
```

## 3. 指标在哪里记录

`RequestMetrics` 使用单调时钟记录以下边界：

```text
arrival_time          请求进入 generate/add_request
first_scheduled_time  第一次被调度
first_token_time      第一个输出 token 回到 host
finish_time           最后一个输出 token 回到 host
```

派生指标：

```text
TTFT = first_token_time - arrival_time
TPOT = (finish_time - first_token_time) / (output_tokens - 1)
E2E  = finish_time - arrival_time
```

这里测量的是用户可观察的 host wall time，不是单个 CUDA kernel 的时间。
一次 batched model step 的耗时会记到该 batch 中每个请求上，因为每个请求
都真实等待了这段时间。

TPOT 是所有 token 间隔的平均值，可能掩盖一次很长的卡顿。因此项目还记录
每个请求的 `inter_token_gap_p95_ms` 和 `max_inter_token_gap_ms`。调度器内部
的 `decode_ms` 只累计真正执行模型的时间，不包含中间为其他请求执行
prefill 的墙钟等待。

## 4. 原调度与 SLO 调度

`prefill_first` 保留上游行为：只要 waiting queue 中还有请求，就优先执行
prefill。它有利于尽快接纳新请求，但可能阻塞已经开始 decode 的请求。

`slo_aware` 保留为 v1 消融基线，使用四个规则：

1. prefill 后至少执行一次 decode，避免长 prompt 连续阻塞输出。
2. 连续 decode 达到上限后推进 waiting queue，避免新请求饿死。
3. TTFT 未逼近目标时只插入一个 KV block 的 prefill；超过目标后使用完整
   chunk，尽快追赶首 token。
4. decode 完成后将 sequence 放回队尾，通过 round-robin 避免尾部请求饿死。

这不是“永远更快”的策略，而是可调的延迟权衡。项目实验要证明它在哪些
工作负载下改善 P95 TTFT 或 P95 TPOT，以及付出了多少吞吐代价。

`slo_aware_v2` 针对 v1 的负实验结果增加三项机制：

1. 第一次 prefill 后立即得到成本样本，再继续准入请求，直到 decode 的
   deadline 更紧迫。初始 batch 大小因此由实测成本和 SLO 决定，而不是固定
   为 1 或 `max_num_seqs`。
2. waiting 请求使用 `arrival + TTFT SLO` 作为 deadline，running 请求使用
   `last_token + TPOT SLO` 作为 deadline。两者都减去预测执行成本，再用
   “剩余时间 / SLO 目标”归一化；值更小的一方更紧迫。
3. Scheduler 对实测 prefill 每 token 时间和 decode step 时间维护 EWMA。
   只有 deadline 允许时才缩小 chunk，并按 KV block 对齐；TTFT 已超期时
   使用完整 chunk 追赶，而不是固定拆成 256 token。

如果 TTFT 与 TPOT 同时超期，调度器选择归一化超期更严重的一方。这不能在
过载时凭空满足所有目标，但能让决策和失败原因可解释。

## 5. 离线批处理与在线到达

一次 `LLM.generate(prompts)` 会先提交所有 prompt，属于 bulk arrival。
这种负载通常有利于 `prefill_first`，不能代表真实服务中“decode 期间不断有
新请求到达”的情况。

`benchmark_slo.py` 支持三种到达模式：

```text
bulk      所有请求在同一时刻到达
constant  按固定 request rate 到达
poisson   到达间隔服从指数分布，模拟无记忆请求流
```

在线模式通过 `LLMEngine.add_request` 和 `step` 驱动，同一随机种子会生成相同
prompt 和到达时间。比较策略时必须固定 arrival pattern、request rate 和
seed。

## 6. Paged KV cache 与 prefix cache

`BlockManager` 将 KV cache 切成固定大小的 block。Sequence 保存逻辑
`block_table`，Attention 根据它找到物理 cache block。

完成的整块 prompt 会按“前一块 hash + 当前 token”生成链式 hash。新请求
拥有相同前缀时，可以直接引用已经存在的 cache block。实验时共享前缀至少
要覆盖一个完整 block；默认 block size 是 256 token。

prefix cache 命中率是“复用的 prompt token / prompt token”，不是请求命中数。

## 7. Triton KV 写入 kernel

`store_kvcache_kernel` 的每个 Triton program 负责一个输入 token：

1. 从 `slot_mapping` 读取该 token 的物理 cache 位置。
2. 合并 KV heads 与 head dimension 为连续的 `D`。
3. 合并读取 key/value，再写入 K/V cache。

kernel 使用 `next_power_of_2(D)` 决定 Triton block 宽度，再通过 `offset < D`
mask 保护尾部。这避免了 `D` 不是 2 的幂时 `tl.arange` 无法编译，同时对
Qwen3-8B 的 `D=1024` 不增加 padding。

先运行 `benchmarks/kernels/benchmark_store_kvcache.py`。它会做逐元素正确性
检查，并报告小 batch（decode）和大 batch（prefill）下的耗时及有效带宽。
只有实测后才应该修改 `num_warps`、program 粒度或向量化方式。

## 8. RMSNorm 与残差融合

RMSNorm 对每一行最后一维计算：

```text
y = x * rsqrt(mean(x^2) + eps) * weight
```

这个算子包含 reduction，但通常更受显存读写影响。
`Add+RMSNorm` 在同一个 kernel 中先完成 `x + residual`，同时输出
新 residual 和归一化结果，避免中间张量在多个 kernel 之间往返显存。

Qwen3 需要分开看两类形状：

* `hidden_size=128`：Q/K Norm，row 数还要乘以 attention heads。
* `hidden_size=4096`：模型主干的 input/post-attention/final Norm。

Q/K 是从融合 QKV 投影结果中 `split` 出来的，最后一维连续但
整个张量不是 contiguous。Kernel 必须使用真实 batch stride；在纯连续
微基准上正确，不代表能直接接入模型。

`benchmarks/kernels/benchmark_rmsnorm.py` 同时比较 eager PyTorch、
`torch.compile` 和 Triton。默认仍使用原来的 `torch.compile`；只有
Triton 在真实形状上稳定胜出，并通过端到端 A/B，才应该将它
作为优化后端。

## 9. 推荐复盘问题

完成实验后，应该能独立回答：

1. 第一个 token 为什么在最后一个 prefill chunk 后产生？
2. 为什么 batch 增大通常提高吞吐，但不一定降低延迟？
3. 为什么长 prefill 会恶化正在运行请求的 TPOT？
4. prefix cache 为什么只能稳定复用完整 block？
5. CUDA Graph 为什么主要优化 decode，而不是动态长度的 prefill？
6. kernel 更快时，为什么端到端吞吐可能几乎不变？
7. 为什么 decode kernel P95 下降时，用户看到的 TPOT 仍可能上升？
8. 为什么 bulk workload 不能单独证明在线调度策略有效？
9. Add+RMSNorm 融合节省的主要是 FLOPs 还是显存读写？
10. 为什么 Triton RMSNorm 胜过 eager 仍不足以证明它值得替换
    `torch.compile`？
