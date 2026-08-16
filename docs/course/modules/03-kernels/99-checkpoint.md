# 模块 3 checkpoint

你得到两个候选优化：

| 候选 | 真实 decode shape | Baseline | Optimized | 每 step 调用 |
|---|---:|---:|---:|---:|
| A: KV store | batch 1 | 11us | 4us | 36 |
| B: Add+RMSNorm | batch 1 | 4.75us | 3.82us | 72 |

系统 TPOT 为 17.4ms。独立完成：

1. 分别估算每个 step 的理论节省和 TPOT 占比；
2. 仅根据此表先选择哪个做端到端 A/B，并说明原因；
3. 如果 B 在 batch 8 反而慢 8%，你是否会做全局替换；
4. A 的端到端实测改善 1.52%，B 的变化为 +0.35%，如何写项目结论；
5. 指出“optimized kernel 更快”这句话的 activation cue 与边界。

[Checkpoint 答案与反馈](../../answers/03-kernels/99-checkpoint.md)

