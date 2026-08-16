# 模块 2 checkpoint

使用缩小后的教学参数：

- `now=20s`，TTFT target 400ms，TPOT target 80ms；
- prefill estimate `0.4ms/token`，decode estimate 12ms；
- waiting A：arrival 19.75s，剩余 512 token；
- waiting B：arrival 19.60s，剩余 128 token；
- running：last token 19.95s；
- block size 16，最大 chunk 128。

独立完成：

1. 选择最紧急的 waiting 请求；
2. 比较它与 running 的 normalized slack；
3. 决定 prefill/decode；
4. 若 prefill，计算 block-aligned token budget；
5. 指出 EWMA 在 mixed-length/batch workload 下的一种失效方式，以及你会
   切换到什么更细的成本模型。

[Checkpoint 答案与反馈](../../answers/02-scheduler/99-checkpoint.md)

