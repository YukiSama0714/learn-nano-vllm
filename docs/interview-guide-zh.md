# learn-nano-vllm 面试讲稿

这份讲稿的目标不是背诵结果，而是帮助你在面试中独立解释问题、机制、
证据和边界。所有数字都来自
[SLO-aware v2 实验记录](slo-v2-results-zh.md)。

## 1. 三分钟版本

我基于一个约 1200 行的 nano-vLLM 教学引擎，做了一个面向在线推理的
SLO-aware 调度项目。原始 `prefill_first` 策略会优先处理等待中的 prompt，
TTFT 很低，但长 prefill 会阻塞正在运行的 decode，造成用户看到的流式卡顿。

我先补齐了请求级可观测性，包括 TTFT、TPOT、queue time、prefill chunks、
ITL、Max ITL、prefix-cache hit 和 preemption。然后实现了三种可对照策略：
原始 baseline、失败的固定小 chunk v1，以及基于 deadline slack 和在线成本
估计的 v2。

v1 是一个重要的负实验：固定把 prompt 切成 256-token chunk 后，调度和
kernel launch 开销增加，TTFT、E2E 和吞吐都变差。v2 不固定 chunk，而是
比较 waiting 请求的 TTFT slack 和 running 请求的 TPOT slack，再用 EWMA
估计的 prefill cost 决定当前安全的 chunk 大小。

在 Qwen3-8B、RTX 5090 的 held-out 320 请求上，v2 将 Max ITL P95 从
259.21ms 降到 68.23ms，将至少出现一次 75ms 卡顿的请求从 93.8% 降到
0.6%。吞吐只下降 0.16%，但 E2E P95 增加 8%，这是我明确接受并记录的
流式平滑度权衡。

我还做了两条 kernel 路线。KV-store Triton kernel 的微基准比 PyTorch
快 1.73--2.90 倍，按 36 层预测每步节省 0.252ms，端到端实测 TPOT
节省 0.27ms。RMSNorm Triton 虽然比 eager 快很多，但没有稳定胜过原有
`torch.compile`，端到端也没有收益，所以我保留实验后端但不改默认实现。

这个项目最重要的产出不是某个最大 tokens/s，而是从请求生命周期、调度
机制、GPU 微基准到端到端指标的一条可复现证据链。

## 2. 十五分钟展开顺序

### 2.1 先画请求生命周期

按下面顺序讲，不要从 scheduler 类名开始：

```text
arrival
  -> waiting / queue
  -> one or more prefill chunks
  -> first token
  -> repeated decode steps
  -> finish
```

然后把指标放到时间线上：

- TTFT：arrival 到 first token；
- TPOT：first token 到 finish 的平均 token 间隔；
- Max ITL：单请求经历的最大相邻 token 间隔；
- queue time：请求处于等待状态的累计时间；
- E2E：arrival 到 finish。

关键判断：平均 TPOT 可能很好，但一次很长的 prefill 仍会制造数百毫秒的
Max ITL，所以流式体验必须看 tail。

### 2.2 解释 baseline 与 v1 为什么失败

`prefill_first` 的优势是 prompt 准入快，缺点是 running decode 可能被
长 prefill 饿死。v1 尝试交错 prefill/decode，但使用固定小 chunk：

- 一个 prompt 需要更多调度轮次；
- 增加 kernel launch 和 host 调度开销；
- 后续 waiting 请求更晚进入 running；
- “更公平”没有自动转化成更低的 TTFT 或 E2E。

不要把 v1 隐藏掉。它证明了“chunked prefill”是机制，不是自动优化。

### 2.3 解释 v2 的决策

waiting 请求近似计算：

```text
slack = arrival + TTFT_target - now - predicted_prefill_cost
normalized_slack = slack / TTFT_target
```

running 请求近似计算：

```text
slack = last_token + TPOT_target - now - predicted_decode_cost
normalized_slack = slack / TPOT_target
```

选择 normalized slack 更小的一侧。waiting 队列内部再选择 least-laxity
请求。prefill cost 和 decode step cost 由运行时 EWMA 更新。

动态 chunk 的核心不是“越小越好”，而是：

```text
safe_tokens = positive_decode_slack / prefill_seconds_per_token
```

再按 KV block 对齐，并受最大 chunk 限制。

### 2.4 用结果证明机制

按证据强度讲：

1. 单元测试证明 deadline、least-laxity、动态 chunk 和 round-robin 行为；
2. 0.6B load sweep 证明 baseline 的 Max ITL 随负载快速恶化；
3. 8B SLO sweep 用固定验收阈值选择 75ms，而不是用配置目标移动球门；
4. 新 seed 的 held-out 320 请求验证不是明显过拟合；
5. prefix-cache 实验说明减少 prefill work 会同时改善 TTFT、queue 和 E2E。

### 2.5 用两个 kernel 说明 Amdahl 定律

KV-store 的故事是正例：

```text
7.011us per layer * 36 layers = 0.2524ms predicted saving
0.27ms measured TPOT saving
```

RMSNorm 的故事是反例：

- 对 eager 的倍数很好看；
- 真正 baseline 是已有的 `torch.compile`；
- decode 关键形状胜负混合；
- 加权预测只有约 0.02ms；
- 端到端变化落在噪声内。

### 2.6 主动说边界

- 只验证单张 RTX 5090 和 Qwen3；
- 没有 HTTP、tokenization 和真实流量；
- offered-load tok/s 不是峰值吞吐；
- mixed-length、长上下文和多卡没有同等证据；
- SLO 目标来自实验选择，不代表生产业务目标。

主动说边界会提高可信度，而不是削弱项目。

## 3. 简历 bullet

可以根据岗位选择三条，不要把全部数字塞进一条：

- 基于 nano-vLLM 实现 deadline-driven SLO-aware 调度器，结合 TTFT/TPOT
  normalized slack、least-laxity-first、EWMA 成本估计与动态 chunked
  prefill；在 RTX 5090 + Qwen3-8B 的 320 请求 held-out 实验中，将
  Max ITL P95 降低 73.7%，吞吐下降 0.16%。

- 构建支持 bulk/constant/Poisson arrivals 的请求级推理 benchmark，
  采集 TTFT、TPOT、Max ITL、queue、prefix-cache hit、吞吐和显存，并通过
  固定验收阈值、SLO sweep 与新随机种子验证调度策略。

- 扩展 Triton KV-cache store 以支持非 2 的幂宽度，完成逐元素验证、
  1--4096 token 微基准和端到端 A/B；将 0.252ms 理论节省与 0.27ms
  实测 TPOT 改善对应，并保留无端到端收益的 RMSNorm 负实验。

## 4. 高频追问

### 为什么不用 TPOT P95 代替 Max ITL？

TPOT 是 first token 到 finish 的平均间隔。一次 300ms 卡顿可能被后续很多
正常 token 稀释。Max ITL 直接回答“这个请求最糟糕的一次卡顿有多长”。

### 为什么 normalized slack，而不是直接比较毫秒？

TTFT 和 TPOT 的目标尺度不同。直接比较绝对毫秒会让较大的 TTFT budget
天然占优；归一化后比较的是各自预算被消耗的比例。

### 为什么 waiting 请求里长 prompt 可能先于短 prompt？

成本预测会从 deadline 中扣除剩余 prefill work。相同 arrival 下，长请求
需要更早开始才能满足相同 TTFT deadline。这不是 shortest-job-first，而是
least-laxity-first。

### 为什么 chunk 要按 block 对齐？

KV cache 以固定 block 管理，prefix cache 也只稳定复用完整块。按 block
对齐可以避免调度粒度与缓存管理粒度不一致。最小 chunk 仍会保证一个 block，
所以特别紧的 decode slack 不是绝对保证。

### 为什么 50ms SLO 反而不如 75ms？

50ms 触发更激进的 4-chunk prefill，额外调度和 launch 开销使实际 Max ITL
更高。SLO 是控制参数，不是越小越好；必须用固定外部阈值验收。

### prefix cache 的 0.50 hit rate 是什么意思？

1024-token prompt 中复用了两个 256-token 完整 block，即 512/1024，
不是“50% 的请求命中”。

### 为什么 KV kernel 快 2--3 倍，吞吐几乎不变？

单次 kernel 只有几微秒。它每层都会调用，所以 TPOT 能看到约 1.5% 收益，
但 GEMM、attention、调度和其他 kernel 仍占绝大多数时间。

### 为什么保留 RMSNorm Triton 代码？

它提供真实 stride 支持、融合 Add+RMSNorm 和可复现实验后端，并证明大
prefill 形状有带宽优势。但默认仍是 compiled，保留它不等于宣称生产收益。

### 为什么 Poisson 比 bulk 更适合证明在线调度？

bulk 同时提交所有请求，天然有利于吞吐型批处理。Poisson arrivals 会让
新 prefill 与已有 decode 相遇，才能暴露 decode starvation。

### 如果换成混合长度 workload，会发生什么？

当前 EWMA 将不同 batch size 和长度压缩成一个平均成本，可能低估长 prompt
或高估短 prompt。下一步应按 prefill/decode、batch size 或 token bucket
建模，并用 held-out mixed-length workload 验证。

### 多卡时最可能新增什么瓶颈？

Tensor Parallelism 会加入 AllReduce/AllGather 和同步尾延迟。单卡 slack
模型只能观察总 step time，不能区分计算与通信，也没有证明 PD 分离或通信
overlap 的收益。

## 5. 面试中的停止规则

出现以下情况时不要继续夸大：

- 只看到 microbenchmark，没有端到端 A/B；
- 只看到单个 seed，没有 held-out；
- 违反率阈值跟着配置目标变化；
- offered load 限制了吞吐，却声称峰值吞吐提升；
- 新策略改善 tail，但隐藏 E2E、TTFT 或显存代价。

面试官真正需要确认的是：你能否识别这些边界，并设计下一项最便宜的验证。

