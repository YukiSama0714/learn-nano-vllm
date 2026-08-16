# 最终 checkpoint：新 workload 设计题

这是课程的独立证明，不要先看答案。

## 场景

你有一张 32GB GPU 和一个 8B 模型。线上请求分布变为：

- 50% prompt 128 token；
- 40% prompt 1024 token；
- 10% prompt 8192 token；
- output 固定 128 token；
- arrival 为 2 秒 20 req/s burst，随后 8 秒 idle；
- TTFT SLO 400ms，Max ITL 验收阈值 80ms。

当前 v2 只有全局 prefill ms/token 与 decode step EWMA。

## 交付

在一页内完成：

1. 画出 burst 中 waiting/running 的关键生命周期；
2. 写出当前 v2 最可能的两个失败模式；
3. 提出一个最小 scheduler 修改，不得同时改三个机制；
4. 写出 baseline、ablation、tuning 和 held-out；
5. 列出主指标、护栏、显存与固定违反率阈值；
6. 预测至少一个 tradeoff；
7. 写清采用、拒绝与继续收集数据的条件；
8. 选择一个 kernel microbenchmark，并说明为什么它不能单独支持系统主张。

## 无稿讲解

完成书面答案后，录制一次 15 分钟讲解：

- 3 分钟生命周期和问题；
- 5 分钟机制与代码入口；
- 4 分钟实验；
- 3 分钟边界与下一步。

用 [评分与参考答案](../../answers/04-evidence/99-checkpoint.md) 自评。只有在
录制结束后再查看。

