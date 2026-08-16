# 模块 1 checkpoint 答案

时间线可以写成：

```text
arrival
  -> queue 40
  -> prefill 12
  -> queue 30
  -> prefill 10 + token 1
  -> 18 -> token 2
  -> 22 -> token 3
  -> 90 -> token 4 / finish
```

- TTFT：`40 + 12 + 30 + 10 = 92ms`。
- TPOT：`(18 + 22 + 90) / 3 = 43.33ms`。
- Max ITL：`90ms`。
- Cache-hit rate：`512 / 1024 = 0.50`。

Max ITL 最直接暴露 90ms 卡顿。Prefix cache 只减少需要执行的 prefill work；
它不会消除 arrival 排队、decode/prefill 调度冲突或后续 decode 尾延迟。

如果 TTFT 算成 52ms，说明漏掉了 chunk 之间重新进入 waiting 的 30ms，属于
**生命周期边界错误**。如果 cache hit 写成“半数请求命中”，属于
**定义错误**。

