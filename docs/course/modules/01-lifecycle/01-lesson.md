# 模块 1：请求生命周期与指标

## 你要做到什么

给定请求的时间戳和调度事件，独立计算 TTFT、TPOT、Max ITL，并判断哪项
指标最能解释用户感受到的卡顿。

**前置连接（因果）**：你熟悉“函数执行时间”，但在线推理的用户延迟不是
一次函数调用；它由排队、一个或多个 prefill step 和多个 decode step 组合。

## 先尝试

不要先看源码。画出一个 3-token 输出请求的最少事件：

```text
arrival -> ? -> ? -> token 1 -> ? -> token 2 -> ? -> token 3 -> finish
```

在图上标出 TTFT、TPOT 和每个 inter-token gap。

## 最小模型

请求状态从 `WAITING` 开始。scheduler 选择它做 prefill 时，可能只处理一个
chunk；prompt 全部完成后才生成第一个 token，并进入 `RUNNING`。之后每次
decode 为每个被选中的 sequence 追加一个 token。达到 EOS 或 `max_tokens`
后进入 `FINISHED`。

`RequestMetrics` 记录：

```text
TTFT = first_token_time - arrival_time
TPOT = (finish_time - first_token_time) / (completion_tokens - 1)
ITL_i = token_time_i - token_time_(i-1)
Max ITL = max(ITL_i)
E2E = finish_time - arrival_time
```

TPOT 是平均值；Max ITL 是单请求最坏一次流式停顿。二者不能互相替代。

## 按这个顺序读代码

1. `nanovllm/engine/sequence.py`：状态、prompt/completion token 与 block table。
2. `nanovllm/engine/metrics.py`：时间戳何时写入、指标如何计算。
3. `nanovllm/engine/llm_engine.py`：arrival、step 与 finished metrics 的边界。
4. `nanovllm/engine/scheduler.py` 的 `schedule` 和 `postprocess`。

每读一个 `mark_*` 调用，都回答：“这是 host 时间还是 GPU event？它属于
queue、prefill、decode 还是用户可见 token？”

## 练习

一个请求在 `t=0` 到达。两个 prefill chunk 分别在 `0.10--0.15s` 和
`0.20--0.26s` 执行；第二个 chunk 结束时产生 token 1。token 2 和 token 3
分别在 `0.29s` 和 `0.40s` 产生，并在 `0.40s` finish。

计算：

1. TTFT；
2. 两个 ITL；
3. TPOT；
4. Max ITL；
5. 为什么只报告 TPOT 会弱化真实卡顿？

完成后查看
[练习答案](../../answers/01-lifecycle/01-lesson.md)。

## Changed-surface 独立检查

系统 A 的一个请求 token gaps 为 `[20, 20, 140, 20]ms`；系统 B 为
`[50, 50, 50, 50]ms`。两者 TTFT 和输出长度相同。

选择更适合流式聊天的系统，并同时写出：

- 激活你的判断的指标；
- 这个判断依赖的用户体验假设；
- 什么新条件会让你改选另一个系统。

不要只写“A/B 哪个平均值更低”。完成后进入
[模块 checkpoint](99-checkpoint.md)。

