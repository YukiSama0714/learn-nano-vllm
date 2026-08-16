# 模块 4：证据链、边界与迁移答辩

## 你要做到什么

为一个未见过的在线 workload 设计可证伪实验，并在结果不支持优化时停止，
而不是移动阈值或只选择有利指标。

**前置连接（因果组合）**：生命周期决定测什么，调度模型产生预测，kernel
成本模型估算影响；实验设计负责判断这些因果解释是否经得住新数据。

## 先尝试

候选策略把 Max ITL P95 从 300ms 降到 70ms，但 TTFT 从 120ms 增到
190ms，E2E 增加 10%，吞吐下降 0.5%。你会采用吗？

不要直接回答“会/不会”。先列出缺少的业务 SLO、违反率和 workload 信息。

## 最小实验合同

每项主张在运行前写清：

1. **Hypothesis**：机制改变会影响哪个指标，方向是什么；
2. **Controls**：模型、prompt/output、arrival、seed、batch、后端和环境；
3. **Measurement**：请求级 tail、违反率、吞吐、显存以及原始 JSON；
4. **Decision rule**：固定验收阈值、允许的代价和停止条件；
5. **Validation**：调参数据与新 seed/新 workload held-out 分离。

配置中的 `tpot_slo_ms` 会改变调度行为；评估中的固定 `Max ITL > 75ms`
用于比较不同配置。二者如果绑在一起，调大目标会机械地降低“违反率”，属于
移动球门。

## 三类常见误判

### Workload mismatch

Bulk 同时到达适合测饱和批处理，却不能单独证明在线 arrival 下的 prefill/
decode 竞争。Poisson 更接近持续到达，但也不等于突发或真实 trace。

### Metric mismatch

Offered load 较低时，output tok/s 主要由请求到达速度决定，不是硬件峰值。
流式聊天要看 Max ITL；离线任务可能更看总吞吐和 E2E。

### Selection bias

在 seed 2026 上调 SLO 后仍用同一 seed 报最终数字，会高估泛化。应先冻结
配置，再使用未参与选择的 seed，例如 held-out 4242。

## 练习：补全实验

你怀疑 mixed-length workload 会让单一 prefill EWMA 低估长 prompt 成本。
已有 Qwen3-8B、RTX 5090 和 baseline/v2。

写出：

1. 一句可证伪 hypothesis；
2. 至少五个 controls；
3. workload 长度分布和 arrival pattern；
4. 主指标、护栏指标和固定阈值；
5. tuning/held-out 切分；
6. 支持、否定和“证据不足”分别长什么样。

[练习答案](../../answers/04-evidence/01-lesson.md)

## Changed-surface：口头答辩

不用项目原数字，给自己 5 分钟准备以下问题：

> 如果线上流量从 Poisson 变为 2 秒突发、8 秒空闲，并且 10% 请求是
> 8192-token prompt，你会先改 scheduler，还是先提高
> `max_num_batched_tokens`？

回答必须包含 activation cue、需要的观测、一个预测和 stop/switch condition。
完成后进行 [最终 checkpoint](99-checkpoint.md)。

