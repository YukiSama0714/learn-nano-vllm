# 模块 2 练习答案

waiting：

```text
predicted = 200 * 0.5ms = 100ms
slack = 9.700 + 0.500 - 10.000 - 0.100 = 0.100s
normalized = 0.100 / 0.500 = 0.20
```

running：

```text
slack = 9.960 + 0.075 - 10.000 - 0.015 = 0.020s
normalized = 0.020 / 0.075 = 0.267
```

waiting 的 normalized slack 更小，因此选择 prefill。安全 token 数：

```text
20ms / 0.5ms = 40 tokens
```

向下按 32 对齐得到 32 token，且位于一个 block 与最大 128 之间。

TPOT target 改为 40ms 后，running slack 变为负数：

```text
9.960 + 0.040 - 10.000 - 0.015 = -0.015s
```

running 会更紧急。激活 cue 是“扣除预计 decode 后已经没有余量”。如果过紧
目标迫使 prefill 被切成大量小 chunk，调度和 launch 开销可能反过来恶化真实
tail；这正是 50ms 目标可能不如 75ms 的机制。

常见错误：

- **缺少前置量**：不扣 predicted cost，直接比较 deadline 剩余时间；
- **选择 cue 错误**：比较绝对 slack 而不是 normalized slack；
- **边界错误**：把 token budget 当成硬实时保证，忽略最小一个 block。

