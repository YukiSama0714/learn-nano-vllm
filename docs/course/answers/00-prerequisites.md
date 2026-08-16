# 起点诊断参考

1. TTFT 覆盖 arrival 到 first token；TPOT 是 first token 后的平均 token
   间隔；Max ITL 是单请求最大的相邻 token 间隔。
2. 不能只看队列名称，应比较 waiting TTFT 与 running TPOT 在扣除预测成本后
   的 normalized slack。
3. 端到端还包含多层调用、GEMM、attention、其他 kernel、调度和排队；局部
   speedup 必须乘调用次数再除以系统时间。
4. 正确性失败、真实 baseline 不改善、收益低于噪声、护栏回退或 held-out
   不复现，都应停止采用或切换方法。

路由建议：

- 第 1 题不稳：模块 1 多做一次生命周期图；
- 第 2 题不稳：模块 2 必须完成两次 slack 手算；
- 第 3 题不稳：模块 3 先做 micro-to-macro 估算；
- 第 4 题不稳：模块 4 的 decision rule 不得跳过。
