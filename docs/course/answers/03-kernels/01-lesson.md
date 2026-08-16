# 模块 3 练习答案

开场估算题：每层节省 8us，36 层共节省 288us，即 0.288ms；相对 18ms
TPOT 的理论比例是 1.6%。仍需要真实 shape 正确性、调用次数确认和固定其他
变量的端到端 A/B。

共享前缀 900 token 只能覆盖三个完整 256-token block：

```text
cached_tokens = floor(900 / 256) * 256 = 768
cache_hit_rate = 768 / 1536 = 0.50
```

第四个 block 只有前 132 token 相同，整个 block 不匹配，因此不能复用。

```text
D = 6 * 128 = 768
BLOCK_SIZE = next_power_of_2(768) = 1024
masked_fraction = (1024 - 768) / 1024 = 25%
```

25% mask 不等于慢 25%。小 token 数可能由 launch latency 主导；大 token 数
还取决于编译器是否消除 masked memory access、占用率、寄存器和实际带宽。
必须比较相同 GPU 上的实测 latency/effective bandwidth。

Changed-surface：

```text
saving = (10us - 6us) * 48 = 192us = 0.192ms
share = 0.192 / 24 = 0.8%
```

可选择包含真实 decode batch 分布的在线 A/B，并固定模型、seed、arrival 与
scheduler。示例采用阈值：多个重复中 TPOT 改善超过 0.5%，且 E2E/SLO/显存
无不可接受回退。停止条件：当前 kernel 的总占比已经低于测量噪声，或继续
调参的理论上限小于约定阈值。

阈值不是源码事实；它必须由项目成本和测量稳定性预先规定。
