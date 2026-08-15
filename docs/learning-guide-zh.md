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

## 4. 原调度与 SLO 调度

`prefill_first` 保留上游行为：只要 waiting queue 中还有请求，就优先执行
prefill。它有利于尽快接纳新请求，但可能阻塞已经开始 decode 的请求。

`slo_aware` 使用四个规则：

1. prefill 后至少执行一次 decode，避免长 prompt 连续阻塞输出。
2. 连续 decode 达到上限后推进 waiting queue，避免新请求饿死。
3. TTFT 未逼近目标时只插入一个 KV block 的 prefill；超过目标后使用完整
   chunk，尽快追赶首 token。
4. decode 完成后将 sequence 放回队尾，通过 round-robin 避免尾部请求饿死。

这不是“永远更快”的策略，而是可调的延迟权衡。项目实验要证明它在哪些
工作负载下改善 P95 TTFT 或 P95 TPOT，以及付出了多少吞吐代价。

## 5. Paged KV cache 与 prefix cache

`BlockManager` 将 KV cache 切成固定大小的 block。Sequence 保存逻辑
`block_table`，Attention 根据它找到物理 cache block。

完成的整块 prompt 会按“前一块 hash + 当前 token”生成链式 hash。新请求
拥有相同前缀时，可以直接引用已经存在的 cache block。实验时共享前缀至少
要覆盖一个完整 block；默认 block size 是 256 token。

prefix cache 命中率是“复用的 prompt token / prompt token”，不是请求命中数。

## 6. Triton KV 写入 kernel

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

## 7. 推荐复盘问题

完成实验后，应该能独立回答：

1. 第一个 token 为什么在最后一个 prefill chunk 后产生？
2. 为什么 batch 增大通常提高吞吐，但不一定降低延迟？
3. 为什么长 prefill 会恶化正在运行请求的 TPOT？
4. prefix cache 为什么只能稳定复用完整 block？
5. CUDA Graph 为什么主要优化 decode，而不是动态长度的 prefill？
6. kernel 更快时，为什么端到端吞吐可能几乎不变？
