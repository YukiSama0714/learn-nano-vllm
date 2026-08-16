# 模块 3：KV cache、prefix cache 与 kernel

## 你要做到什么

从逻辑 token 前缀追到物理 KV slot，再把单次 kernel 延迟换算成整模型
TPOT 占比，判断是否值得继续优化。

**前置连接（组合）**：生命周期告诉你 kernel 在 prefill/decode 的哪个位置
发生；Paged KV 决定写到哪里；Amdahl 定律决定局部加速能否影响请求指标。

## 先尝试

一个 kernel 从 12us 优化到 4us，看起来快 3 倍。模型有 36 层，每个 decode
step 每层调用一次，实测 TPOT 为 18ms。

在看后文前估算：

1. 每个 step 节省多少；
2. TPOT 理论改善比例；
3. 还缺什么证据才能说“端到端有效”。

## Paged KV 的最小模型

`Sequence.block_table` 保存逻辑 block 到物理 block id 的映射。decode 新 token
的 slot 近似为：

```text
physical_slot = physical_block_id * block_size + offset_in_block
```

`store_kvcache` 将 K/V heads 与 head_dim 展平成 `D`，读取 `slot_mapping`，
把该 token 的 K/V 写入对应物理 cache。

Prefix cache 只 hash 完整 block，并把“前一个 block hash + 当前 block token”
组成链式 hash。这意味着：

- 部分 block 不稳定复用；
- 相同局部 token 但不同前缀不会错误共享；
- ref count 允许多个 sequence 引用同一个物理 block；
- block 被回收复用时必须清理旧 hash 映射。

## Kernel 的两层验收

第一层是算子：

- 与 PyTorch reference 逐元素一致或在明确 dtype tolerance 内；
- 覆盖真实 shape、stride、边界和非 2 的幂；
- 报告 latency 与 effective bandwidth。

第二层是系统：

```text
predicted_step_saving
  = saving_per_call * calls_per_layer * num_layers

predicted_TPOT_share
  = predicted_step_saving / measured_TPOT
```

然后固定其他变量做端到端 A/B。微基准只能生成预测，不能替代第二层。

## 代码走读

1. `nanovllm/engine/block_manager.py`：allocate、hash、ref count、deallocate。
2. `nanovllm/engine/model_runner.py`：slot mapping 与 block tables 的构造。
3. `nanovllm/layers/attention.py`：KV 写入和 prefill/decode attention 分支。
4. `nanovllm/layers/layernorm.py`：reduction、融合 residual 与真实 stride。
5. `benchmarks/kernels/`：reference、正确性、warmup、repetition、JSON。

特别检查 Q/K 从 QKV `split` 后不是整体 contiguous；只在连续假数据上通过的
RMSNorm kernel 不能直接接入模型。

## 练习

一个 1536-token prompt 与缓存请求共享前 900 token，block size 为 256。
另一个模型的 KV heads=6，head_dim=128。

计算：

1. 最多稳定复用多少完整 token，cache-hit rate 是多少；
2. `D` 是多少，`next_power_of_2(D)` 是多少；
3. masked lane 比例；
4. 为什么 masked lane 比例不能直接等同于实际性能损失。

[练习答案](../../answers/03-kernels/01-lesson.md)

## Changed-surface 独立检查

新模型有 48 层。某 kernel 从 10us 降到 6us，每层 decode 调用一次，系统
TPOT 为 24ms。请给出：

- 理论 TPOT 改善比例；
- 你会选择的端到端 workload；
- 一个采用优化的阈值；
- 一个停止继续手调的条件。

完成后进入 [模块 checkpoint](99-checkpoint.md)。

