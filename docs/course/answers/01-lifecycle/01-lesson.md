# 模块 1 练习答案

最少时间线应包含 arrival、一次或多次 waiting/prefill、first token、两个
decode gap 和 finish。TTFT 从 arrival 覆盖到 token 1；ITL 只存在于相邻
输出 token 之间。

## 时间计算

- TTFT：`0.26 - 0 = 260ms`。
- ITL：`0.29 - 0.26 = 30ms`，`0.40 - 0.29 = 110ms`。
- TPOT：`(0.40 - 0.26) / (3 - 1) = 70ms`。
- Max ITL：`110ms`。

TPOT 将 30ms 和 110ms 平均为 70ms，不能直接表达请求经历过一次 110ms
停顿。若把输出延长，许多正常 gap 会进一步稀释这次卡顿。

## Changed-surface 判断

在“流式聊天最怕偶发停顿”的假设下，B 更合适：A 的 Max ITL 是 140ms，
B 是 50ms。激活 cue 是 tail gap，而不是平均 gap。

如果场景改为后台批量生成、用户看不到逐 token 输出，并且 A 的 E2E 或吞吐
显著更好，就应切换判断指标，可能选择 A。

## 常见因果缺口

- **表面匹配错误**：只比较四个数的均值，没有关联流式体验。
- **关系错误**：把 TPOT 当成每个 token 都经历的固定时间。
- **边界错误**：任何场景都选低 Max ITL，没有说明后台离线任务可能更看重
  E2E 或吞吐。
