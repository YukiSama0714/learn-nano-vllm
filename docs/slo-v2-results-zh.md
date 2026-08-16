# SLO-aware v2：RTX 5090 实验结果

本页记录从失败的 v1 到 held-out 验证的完整证据链。原始 JSON 和日志保留在
服务器的 `runs/` 目录，不提交到 Git；复现参数见
[`rtx5090-experiments.md`](rtx5090-experiments.md)。

## 1. 环境与固定条件

```text
GPU                 NVIDIA GeForce RTX 5090 32GB
PyTorch             2.8.0+cu128
Triton              3.4.0
FlashAttention      2.8.3
模型                Qwen3-0.6B / Qwen3-8B
输入/输出长度       1024 / 128 tokens
max_num_seqs        8
prefill_chunk_size  1024
TTFT SLO            500ms
```

请求指标使用 host 单调时钟。`Max ITL` 是单个请求经历的最大 token 间隔，
“ITL 违反率”是至少一次超过固定阈值的请求比例。

## 2. v1 负实验

0.6B、32 个请求同时到达时，v1 将 prompt 固定拆成 256-token chunk：

| 指标 | prefill_first | slo_aware v1 |
|---|---:|---:|
| TTFT P95 ms | 602.08 | 1101.29 |
| TPOT P95 ms | 11.67 | 13.23 |
| Queue P95 ms | 582.85 | 1081.60 |
| Chunks P95 | 1 | 4 |
| E2E P95 ms | 2061.78 | 2351.17 |
| Output tok/s | 2064.04 | 1765.36 |

结论：交错 prefill/decode 本身不能保证低延迟。固定小 chunk 增加调度和
kernel launch 开销，同时延迟了后续请求准入。

## 3. 0.6B 在线负载扫描

Poisson arrival、每个负载点 64 请求、重复 3 次、seed 2026：

| Req/s | Baseline Max ITL P95 ms | v2 Max ITL P95 ms | Baseline ITL violations | v2 ITL violations | Baseline tok/s | v2 tok/s | v2 TTFT P95 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 42.34 | 24.39 | 1.6% | 1.0% | 564.08 | 564.09 | 46.17 |
| 8 | 125.07 | 25.90 | 12.0% | 1.0% | 1104.97 | 1105.91 | 51.18 |
| 12 | 365.35 | 28.34 | 51.6% | 1.0% | 1589.61 | 1595.62 | 68.25 |
| 16 | 905.71 | 33.27 | 84.9% | 1.0% | 1913.20 | 1932.79 | 112.42 |

`prefill_first` 保持了低 TTFT，但负载升高后 decode starvation 快速恶化。
v2 的 Max ITL P95 在 24–33ms 之间，TTFT 违反率始终为 0%，吞吐没有可见
回退。0.6B 的饱和拐点约在 12–16 req/s。

## 4. 8B 的 SLO 灵敏度

Poisson 2 req/s、32 请求、重复 3 次、seed 2026。所有违反率都使用固定
TTFT 500ms、Max ITL 75ms 验收：

| Policy/TPOT target | TTFT P95 ms | Max ITL P95 ms | Queue P95 ms | Chunks P95 | E2E P95 ms | ITL violations | Output tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| prefill_first | 157.31 | 192.12 | 75.88 | 1 | 2034.77 | 91.7% | 250.80 |
| v2 / 50ms | 370.06 | 70.08 | 253.96 | 4 | 2460.44 | 2.1% | 249.09 |
| v2 / 75ms | 211.59 | 56.36 | 123.41 | 2 | 2103.77 | 2.1% | 250.53 |
| v2 / 100ms | 201.14 | 93.97 | 106.28 | 2 | 2122.80 | 14.6% | 250.26 |
| v2 / 150ms | 168.72 | 94.28 | 86.66 | 1 | 2045.00 | 91.7% | 250.78 |

50ms 目标反而被 75ms 目标严格支配。过严的目标触发 4 个小 chunk，额外
开销使实际 Max ITL 更高。最终按“TTFT 违反率为 0、ITL 违反率不超过 5%，
再最小化 E2E”的规则选择 75ms。

## 5. Held-out 验证

为避免在同一随机负载上调参和报告，最终只比较 baseline 与选定的 v2 配置。
使用新 seed 4242、64 请求、重复 5 次，共 320 个请求：

| 指标 | prefill_first | slo_aware_v2 / 75ms |
|---|---:|---:|
| TTFT P95 ms | 156.27 | 213.56 |
| TTFT violations > 500ms | 0.0% | 0.0% |
| TPOT P95 ms | 16.47 | 17.40 |
| Max ITL P95 ms | 259.21 | 68.23 |
| Requests with Max ITL > 75ms | 93.8% | 0.6% |
| Queue P95 ms | 74.82 | 124.76 |
| Chunks P95 | 1 | 2 |
| E2E P95 ms | 2180.29 | 2353.69 |
| Output tok/s | 255.56 | 255.16 |
| Peak GiB | 26.00 | 26.00 |

在未参与调参的 320 个请求上，v2 将 Max ITL P95 降低 73.7%，将发生
75ms 以上卡顿的请求从约 300 个降到约 2 个；吞吐下降 0.16%，E2E P95
增加 8.0%，TTFT 仍全部满足 500ms 目标。

## 6. 机制解释

v2 不追求每项指标都更小，而是在 TTFT 和流式平滑度之间执行可观测的
deadline 权衡：

1. 对 waiting 请求计算 `arrival + TTFT SLO - predicted prefill cost`。
2. 对 running 请求计算 `last token + TPOT SLO - predicted decode cost`。
3. 比较归一化 slack，选择更紧迫的一方。
4. 用运行时 EWMA 估计成本，并将 prefill chunk 限制在 decode slack 内。

同一代码在 0.6B 上选择 1 个 chunk，在 8B/75ms 上选择 2 个 chunk，说明
行为来自运行时成本而不是写死模型名称。

## 7. KV-cache 写入 kernel

`store_kvcache` 先做逐元素正确性检查，再比较 Triton kernel 与
PyTorch advanced indexing 基线。`D = num_kv_heads * head_dim`：

| D | Tokens | Triton ms | PyTorch ms | Speedup | Triton GB/s |
|---:|---:|---:|---:|---:|---:|
| 768 | 1 | 0.004105 | 0.011189 | 2.725x | 1.50 |
| 768 | 8 | 0.004143 | 0.012031 | 2.904x | 11.86 |
| 768 | 64 | 0.004223 | 0.012206 | 2.890x | 93.11 |
| 768 | 512 | 0.006130 | 0.015105 | 2.464x | 513.21 |
| 768 | 4096 | 0.020120 | 0.037264 | 1.852x | 1250.79 |
| 1024 | 1 | 0.004094 | 0.011105 | 2.712x | 2.00 |
| 1024 | 8 | 0.004131 | 0.011931 | 2.888x | 15.86 |
| 1024 | 64 | 0.004169 | 0.012080 | 2.897x | 125.74 |
| 1024 | 512 | 0.006141 | 0.015938 | 2.595x | 682.98 |
| 1024 | 4096 | 0.025800 | 0.044753 | 1.735x | 1300.58 |

1--64 tokens 时 Triton 耗时稳定在约 4.1us，说明这一段主要受 kernel
launch 延迟限制；512--4096 tokens 时开始受显存带宽限制。

Qwen3-8B 的 `D=1024` 不需要 padding。`D=768` 会将 Triton block
扩到 1024 并 mask 掉 25% 的 lane；在 512 tokens 时两种形状的绝对
耗时几乎相同，因此按实际字节数计算的有效带宽低 25%。到
4096 tokens 时，768 和 1024 的有效带宽只相差 3.8%，说明 mask
没有引入灾难性回退。

这些数字是微基准，不等于端到端提速。以单 token、`D=1024`
为例，每次调用只节省约 7us；需要乘以模型层数，再与实测
TPOT 比较，才能估算整体收益。因此本项目将该 kernel 作为
“正确性 + 非 2 的幂形状支持 + naive/optimized 对照”的算子案例，
而不声称它解决了端到端瓶颈。

### 7.1 从微基准预测端到端收益

Qwen3-8B 共有 36 层。根据单 token、`D=1024` 的微基准，
Triton 相对 PyTorch 每层节省 `0.011105 - 0.004094 = 0.007011ms`，
因此预测每个 decode step 节省：

```text
0.007011ms * 36 = 0.2524ms
```

使用相同 Qwen3-8B、Poisson 2 req/s、seed 4242、320 个请求和
`slo_aware_v2 / 75ms`，仅替换 KV 写入实现：

| 指标 | Triton | PyTorch | Triton 变化 |
|---|---:|---:|---:|
| TTFT P95 ms | 214.97 | 212.03 | +1.39% |
| TPOT P95 ms | 17.44 | 17.71 | -1.52% |
| Max ITL P95 ms | 68.03 | 68.78 | -1.09% |
| E2E P95 ms | 2348.45 | 2387.72 | -1.64% |
| Output tok/s | 255.18 | 254.99 | +0.07% |
| Peak GiB | 27.34 | 27.34 | 0.00% |

实测 TPOT 节省 0.27ms，与预测的 0.2524ms 相差约 7%。E2E P95
降低 39.27ms，也与 128-token 生成过程中逐 step 累积的收益数量级
一致。TTFT 的小幅反向变化没有超过系统噪声，两组的 SLO 违反率也
相同。

`Output tok/s` 在 2 req/s 下主要受 offered load 限制，因此不用它
声称峰值吞吐提升。该 A/B 的结论是：Triton kernel 的微观收益可以
在系统 TPOT 中观测，但它只占约 1.5%，不是下一个值得手调的瓶颈。

## 8. 局限性

1. 仅测试单张 RTX 5090 和 Qwen3 模型族。
2. 使用固定长度的随机 token prompt，没有 HTTP、tokenization 或真实流量。
3. Poisson 实验的 tok/s 受 offered load 限制，不代表峰值离线吞吐。
4. EWMA 成本与 batch size、prompt 长度有关，混合长度负载仍需验证。
5. Held-out 有 320 个请求，足以验证方向，但不能替代生产规模压测。
6. Kernel 端到端 A/B 只覆盖一组 8B 在线负载，尚未扫描 batch size
   和 prompt 长度对 kernel 收益的影响。
