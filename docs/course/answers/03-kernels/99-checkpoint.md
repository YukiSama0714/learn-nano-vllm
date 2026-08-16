# 模块 3 checkpoint 答案

候选 A：

```text
(11 - 4)us * 36 = 252us = 0.252ms
0.252 / 17.4 = 1.45%
```

候选 B：

```text
(4.75 - 3.82)us * 72 = 66.96us = 0.06696ms
0.06696 / 17.4 = 0.38%
```

仅凭表格应先验证 A，因为预测贡献更大，且 shape 与调用次数明确。B 在 batch
8 变慢时不能全局替换；至少要按真实 batch 分布加权，或研究有证据的 hybrid
dispatch。不能依据 batch 1 单点选择。

项目结论应写：

- A：micro speedup 可按层数预测，端到端 TPOT 改善 1.52%，与预测同数量级；
- B：部分 shape 有收益，但相对现有 compiled baseline 不稳定，端到端无
  可测收益，因此默认不切换。

activation cue 是“在真实 shape、真实 baseline、正确性通过的条件下，局部
耗时占端到端比例足够大”。边界包括调用次数错误、GPU graph/stride 不同、
workload 改变以及收益低于噪声。

如果只按 2.88x、1.24x 的倍数选候选，属于**表面数字匹配**；缺失的是调用
次数和端到端分母。

