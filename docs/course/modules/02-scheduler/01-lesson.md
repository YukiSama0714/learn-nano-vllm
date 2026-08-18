# 模块 2：SLO-aware 调度决策

## 你要做到什么

面对 waiting prefill 与 running decode 的竞争，计算 normalized slack，
选择下一步，并说明动态 chunk 的停止或失效条件。

**前置连接（类比）**：deadline scheduling 会优先处理剩余余量更小的任务。
这里的类比在“按 laxity 选择”处成立，但 GPU 请求会组成 batch，成本需要
预测，也不能在一个 kernel 中途任意抢占。

## 先尝试

不看公式回答：如果 waiting 请求还有 500 token prompt，距离 TTFT deadline
还有 100ms；running 请求距离下一 token deadline 还有 30ms，你一定应该
decode 吗？写出还缺少的一个信息。

正确方向不是直接比较 100 和 30，而是先扣除各自预计执行成本。

## v2 的最小决策模型

waiting 请求：

```text
predicted_prefill = remaining_tokens * prefill_seconds_per_token
slack = arrival + TTFT_target - now - predicted_prefill
normalized = slack / TTFT_target
```

running 请求：

```text
slack = last_token + TPOT_target - now - predicted_decode_step
normalized = slack / TPOT_target
```

选择 normalized slack 更小的一侧。多个 waiting 请求之间使用 least-laxity；
多个 running 请求通过 deque 轮转。

成本不是常数表，而由最近 model step 的 host duration 更新：

```text
estimate = alpha * sample + (1 - alpha) * estimate
```

当决定执行 prefill 时：

```text
safe_tokens = positive_decode_slack / prefill_seconds_per_token
```

v2 再按 KV block size 向下对齐。v3 使用独立的
`prefill_chunk_granularity`，避免 page16 与 block256 在相同 workload 下产生
不同的调度粒度；它限制在 `[one granularity, prefill_chunk_size]`。

## 手算练习

为了方便手算，假设 block size 为 32，最大 chunk 为 128：

- `now = 10.000s`；
- waiting：arrival `9.700s`，TTFT target `500ms`，剩余 200 token；
- prefill estimate：`0.5ms/token`；
- running：last token `9.960s`，TPOT target `75ms`；
- decode estimate：`15ms/step`。

计算：

1. waiting normalized slack；
2. running normalized slack；
3. 下一步选 prefill 还是 decode；
4. 如果选 prefill，token budget 是多少？

[练习答案](../../answers/02-scheduler/01-lesson.md)

## 读代码时只追四条路径

1. `_waiting_sequence_slack`：deadline 减预测成本；
2. `_decode_slack`：以 last token 为参考；
3. `_should_schedule_prefill_v2`：比较 normalized slack；
4. `_prefill_token_budget_v2`：把 decode slack 转成 token budget。

然后读 `_schedule_prefill` 和 `_schedule_decode`，确认 block allocation、
partial prefill 限制与 round-robin 如何改变理论决策。

## Changed-surface 独立检查

现在把 TPOT target 从 75ms 改为 40ms，但 decode estimate 仍为 15ms。
不重新抄完整公式，说明：

- 哪一侧的 normalized slack 会怎样变化；
- activation cue 是什么；
- 哪个条件会让“更紧的 TPOT target”反而恶化真实延迟。

完成后进入 [模块 checkpoint](99-checkpoint.md)。
