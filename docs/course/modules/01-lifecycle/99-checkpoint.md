# 模块 1 checkpoint

一个 1024-token prompt 有 512 token prefix-cache hit，剩余部分被拆成两个
prefill chunk。请求到达后排队 40ms，两个 chunk 分别执行 12ms 和 10ms，
中间因为 decode 优先又等待 30ms。第一个 token 在第二个 chunk 结束时产生；
后续 3 个 token 的间隔为 `[18, 22, 90]ms`。

不看答案完成：

1. 画出生命周期并区分“执行”和“排队”；
2. 计算 TTFT、TPOT、Max ITL 和 cache-hit rate；
3. 指出哪一个值最能暴露 90ms 卡顿；
4. 说明 prefix hit 为什么没有保证低 Max ITL。

[Checkpoint 答案与反馈](../../answers/01-lifecycle/99-checkpoint.md)

