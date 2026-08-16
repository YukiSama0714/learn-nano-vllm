# 模块 2 checkpoint 答案

waiting A：

```text
predicted = 512 * 0.4ms = 204.8ms
slack = 19.75 + 0.4 - 20 - 0.2048 = -54.8ms
normalized = -0.137
```

waiting B：

```text
predicted = 128 * 0.4ms = 51.2ms
slack = 19.60 + 0.4 - 20 - 0.0512 = -51.2ms
normalized = -0.128
```

A 虽然到达更晚，但剩余工作更多，laxity 更小，因此先选 A。

running：

```text
slack = 19.95 + 0.08 - 20 - 0.012 = 18ms
normalized = 0.225
```

A 已逾期，因此选择 prefill。budget：

```text
18ms / 0.4ms = 45 tokens
floor_to_block(45, 16) = 32 tokens
```

一种失效方式：同一个 prefill ms/token EWMA 混合了 batch 1 与大 batch、
短 prompt 与长 prompt；刚发生负载切换时，它会系统性低估或高估成本。
可切换到按 batch-size/token bucket 分桶的估计，或使用带 batch/length
特征的轻量模型，并在 held-out mixed-length workload 上验收。

若只说“调小 alpha”，属于**表面参数修补**；它改变响应速度，但没有解决
不同形状共享一个均值的关系错误。

