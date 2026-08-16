# 最终 checkpoint 评分与参考

## 参考结构

生命周期应指出：burst 早期大量短/中/长请求进入 waiting；首批请求完成
prefill 后进入 running；后续长 prefill 与已有 decode 竞争；idle 阶段则
几乎没有 TTFT/TPOT 冲突。

两个合理失败模式：

1. 全局 EWMA 被大量短 prompt 拉低，低估 8192-token step，chunk 过大；
2. burst 的 batch size 突变使 decode/prefill cost estimate 滞后，normalized
   slack 排序基于过期成本。

最小修改示例：只将 prefill ms/token 按 scheduled-token bucket 分成
small/medium/large 三个 EWMA；不要同时更换调度目标、队列策略和 preemption。

实验至少包括：

- `prefill_first` baseline；
- 原 v2；
- v2 + bucketed estimator；
- 去掉 bucket 或固定 oracle cost 的 ablation；
- tuning seeds 与新的 burst held-out seeds。

主指标：TTFT > 400ms 和 Max ITL > 80ms 的请求违反率。护栏：E2E P95、
峰值/平均吞吐、peak reserved memory、preemptions 和 OOM。应报告分长度组
指标，避免 10% 长请求被总体平均掩盖。

合理预测：bucket estimator 可能降低长 prompt 引起的 Max ITL，但在新 bucket
样本不足时估计更噪，并可能增加短请求 TTFT 或减少 batch efficiency。

一个可用 decision rule：

- 采用：held-out 中两项 SLO 违反率满足阈值，长请求 Max ITL 明显下降，
  吞吐/E2E/显存不超过预设代价；
- 拒绝：改善只存在 tuning seed、只改善平均 TPOT，或护栏失败；
- 继续收集：置信区间/重复波动覆盖效果，且没有明确方向性回退。

Kernel 部分可以选择 KV-store 或 RMSNorm，但必须把 per-call saving 乘以调用
次数并除以 TPOT/prefill step；microbenchmark 不能解释 scheduler queue 或
burst arrival。

## 评分

每项 0--2 分，共 16 分：

1. 生命周期是否正确；
2. 失败模式是否因果可检验；
3. 修改是否最小；
4. baseline/ablation 是否能归因；
5. 指标与固定阈值是否匹配场景；
6. tradeoff 是否具体；
7. stop/switch condition 是否预先定义；
8. micro-to-macro 是否正确。

- 13--16：可以进行无稿面试；
- 9--12：重做得分最低的一个模块 checkpoint；
- 0--8：先回到模块 1 和 2，不要继续背讲稿。

若答案漏掉 activation cue 或停止条件，即使数字正确，最多 12 分。

