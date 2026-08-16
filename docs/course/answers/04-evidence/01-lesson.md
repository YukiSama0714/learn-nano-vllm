# 模块 4 练习答案

开场表格不足以直接决定是否采用。至少需要 TTFT/Max ITL 的业务阈值、请求
违反率、arrival/长度分布、重复波动和预先约定的 E2E/吞吐护栏。

一个合格 hypothesis：

> 当 prompt 长度呈双峰或长尾分布时，全局 prefill ms/token EWMA 会在负载
> 切换后低估长 prompt step，导致动态 chunk 超过 decode slack，使
> Max ITL 违反率高于按长度/batch 分桶的成本估计。

Controls 至少包括：模型与权重、GPU/软件版本、prompt/output 分布、
arrival trace、随机 seed、`max_num_seqs`、KV block、scheduler 其他参数、
RMS/KV 后端和重复次数。

Workload 可以使用固定比例的 128/1024/8192 prompt，先做 seeded Poisson，
再用 burst trace 做 transfer。主指标是 Max ITL P95 与请求违反率；护栏包括
TTFT 违反率、E2E、output tok/s 和 peak memory。阈值必须在运行前固定。

先在一组 seeds 选择 bucket 边界或 alpha，冻结后用新 seeds 和 burst
workload 验证。

- **支持**：分桶模型在 held-out 中降低 Max ITL 违反率，且 TTFT/吞吐护栏
  满足预设范围；
- **否定**：违反率无改善或代价超过护栏；
- **证据不足**：差异与 run-to-run 波动同量级、样本太少或出现 OOM 导致
  workload 不一致。

常见错误是只写“多跑几次”。重复次数不能修复 workload mismatch 或移动
阈值，属于**证据类型错误**。

Changed-surface 回答没有唯一机制。一个合格判断应先观察 burst 开始时的
waiting/running、cost estimate lag、KV memory 和 Max ITL，再决定是否修改
scheduler。直接提高 batch token 可能增加长 prefill 阻塞和显存；直接改
scheduler 也可能在 idle 阶段损失吞吐。stop condition 应绑定固定 SLO 与
护栏，而不是“数字看起来更好”。
