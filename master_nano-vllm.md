# nano-vllm Master 学习手册：CUDA × LLM 推理引擎

> 面向已经完成 CS336 Assignment 1、理解 Transformer、具备基础 CUDA/C++ 编程能力的学习者。
>
> 目标不是“看懂一个 1200 行仓库”，而是最终能够独立追踪、修改、验证和优化一个带 Paged KV Cache、Continuous Batching、Prefix Cache、CUDA Graph 与 Tensor Parallelism 的最小 LLM 推理引擎。

---

## 快速导航

- [0. 文档基线与使用方法](#section-0)
- [1. 环境准备与五分钟运行](#section-1)
- [2. 开始前的低风险诊断](#section-2)
- [3. nano-vllm 全局地图](#section-3)
- [4. 正确性与性能纪律](#section-4)
- [5. P0：基线与请求追踪](#section-5)
- [6. P1：自定义 CUDA 算子接入](#section-6)
- [7. P2：归约、RMSNorm 与 Sampler](#section-7)
- [8. P3：Paged KV 内存引擎](#section-8)
- [9. P4：Prefill、Decode 与 Roofline](#section-9)
- [10. P5：CUDA Graph 与连续 Attention](#section-10)
- [11. P6：Paged Decode Attention](#section-11)
- [12. 可选分支](#section-12)
- [13. 最终综合项目](#section-13)
- [14. 推荐周计划](#section-14)
- [15. 高频错误与排查](#section-15)
- [16. 关键公式速查](#section-16)
- [17. 自测题](#section-17)
- [18. 推荐资料](#section-18)

如果今天就开始：依次完成 1.1–1.5，做第 2 节诊断，然后进入 P0。不要先通读 2000 行文档，也不要先写最终 attention kernel。

---

<a id="section-0"></a>

## 0. 文档基线与使用方法

### 0.1 版本基线

本文按以下版本编写和核对：

- 官方仓库：[`GeeeekExplorer/nano-vllm`](https://github.com/GeeeekExplorer/nano-vllm)
- 固定提交：[`bb823b3`](https://github.com/GeeeekExplorer/nano-vllm/tree/bb823b3e06983d71485a8e1f23715ebd87d98ef8)
- 提交主题：chunked-prefill refactor
- 核对日期：2026-08-06
- Python：`>=3.10,<3.13`
- 官方依赖下限：PyTorch 2.4、Triton 3.0、Transformers 4.51，另需 flash-attn、xxhash
- 主模型：Qwen3-0.6B，默认路径 `~/huggingface/Qwen3-0.6B`

本文中的文件行号以该提交为准。若你的仓库更新了，优先按函数名搜索，不要机械依赖行号。

### 0.2 已核实的模型形状

本文默认使用以下 Qwen3-0.6B 形状：

| 参数 | 值 |
|---|---:|
| hidden size | 1024 |
| decoder layers | 28 |
| query heads | 16 |
| KV heads | 8 |
| head dim | 128 |
| intermediate size | 3072 |
| vocabulary size | 151936 |
| KV block size | 256 tokens |
| 默认最大上下文 | 4096 tokens |
| 推荐推理 dtype | 以模型配置为准，通常为 BF16 |

由此可得单卡 TP=1 时的主要矩阵形状：

- Q 输出：`16 × 128 = 2048`
- K 输出：`8 × 128 = 1024`
- V 输出：`8 × 128 = 1024`
- QKV 合并输出：`1024 → 4096`
- Attention 输出投影：`2048 → 1024`
- Gate/Up 合并投影：`1024 → 6144`
- Down 投影：`3072 → 1024`

这些常数是本路线的锚，不应成为 kernel 的隐式假设。除明确标注为 Qwen3 快速路径的代码外，每个实验都必须测试至少一个改变后的合成形状。

### 0.3 课程设计基础

**真实问题**：你已经会实现 Transformer，但“能写模型”不等于“能服务模型”。推理引擎需要在请求状态、KV 内存、动态批处理、GPU kernel、CUDA Graph 和通信之间维持一组跨层不变量。

**起点假设**：你能阅读 PyTorch 模型代码，知道 Q/K/V、causal attention、RMSNorm、RoPE、GQA；能写使用 `threadIdx`、`blockIdx`、shared memory、`atomicAdd` 的简单 CUDA kernel，并能用 nvcc 编译。

**最终可观察成果**：在不照抄现成实现的情况下，你能：

1. 从请求进入队列开始，追踪一个 token 经调度、元数据准备、模型前向、采样和状态更新的完整生命周期。
2. 手推任意序列的逻辑 token 到物理 KV slot 的映射，解释 prefix cache、block table、引用计数和重算式抢占。
3. 编写并接入 graph-safe 的 PyTorch CUDA 扩展，正确处理 stream、dtype、尾部、数值累加和边界输入。
4. 用 CUDA Event、Nsight Systems、Nsight Compute 分别回答“多慢、慢在哪里、为什么慢”。
5. 独立实现正确的 paged decode-attention baseline，并说明它与成熟 FlashAttention/Flash-Decoding kernel 的差距。
6. 面对改变后的 head 数、GQA 比例、上下文长度、block size 或 workload，重新判断瓶颈和设计，而不是套固定结论。

**最终证明任务**：在混合 workload（长 prompt、短 decode、共享前缀、不同上下文长度）上，用自写 paged decode-attention 替换 nano-vllm 的 decode attention 路径；证明正确性、Graph 兼容性和性能边界，并写出一份可复现实验报告。

### 0.4 什么属于仓库事实，什么属于教学扩展

**仓库事实**：官方代码中的 API、类、数据流、默认配置和库调用。例如 nano-vllm 当前使用 Triton 写 `store_kvcache`，使用 flash-attn 处理 prefill/decode attention，使用 `torch.compile` 处理若干逐元素/归约算子。

**教学扩展**：本文建议创建的 `labs/`、测试框架、手写 CUDA kernel、性能记录和调度实验。这些不是官方仓库要求，而是为了形成可迁移能力。

**硬件相关假设**：任何“带宽受限”“计算受限”“比库慢几倍”的判断都必须在你的 GPU、dtype 和形状上重新测量。本文给出预测模型，不把硬件相关结论当作定律。

### 0.5 推荐目录

在你自己的 nano-vllm fork 中创建以下目录。不要直接覆盖原实现；所有替换都应有开关，可以随时回到 reference path。

```text
nano-vllm/
├── nanovllm/                 # 官方源码
├── labs/
│   ├── common/
│   │   ├── correctness.py    # allclose、误差统计、随机形状
│   │   ├── benchmark.py      # warmup、CUDA Event、统计输出
│   │   └── shapes.py         # 统一测试形状
│   ├── p0_baseline/
│   ├── p1_silu/
│   ├── p2_reduction/
│   ├── p3_kvcache/
│   ├── p4_roofline/
│   ├── p5_graph_attention/
│   └── p6_paged_attention/
├── reports/
│   ├── environment.md
│   ├── pattern-cards.md
│   └── final-report.md
└── master_nano-vllm.md
```

建议每个阶段单独建分支或 commit。一次只改变一个机制，方便 `git diff` 和性能回归。

### 0.6 每次学习都遵循同一个闭环

每个实验必须按以下顺序进行：

1. **预测**：先写下张量形状、读写字节、预期瓶颈和可能失败的边界。
2. **最小正确实现**：先追求易证明的 baseline，不提前做向量化或融合。
3. **独立测试**：与 PyTorch/Triton/flash-attn 参考对比。
4. **边界测试**：使用非对齐形状、短序列、跨 block、`slot=-1` 等情况。
5. **接入引擎**：用开关在 reference/custom 两条路径间切换。
6. **测量**：先时间线，后 kernel counters；不要拿一次运行结果下结论。
7. **只改一个变量优化**：记录变化前后的时间、字节和原因。
8. **迁移检查**：改变一个表面条件，再次完成预测和验证。

若一个步骤失败，先分类原因：

- 缺少前置知识；
- 索引关系错误；
- 只匹配了固定形状；
- 选错了优化方向；
- 越过了方法边界，例如用 ncu 判断 CPU launch gap。

---

<a id="section-1"></a>

## 1. 环境准备与五分钟运行

### 1.1 硬件和系统检查

必须具备 NVIDIA GPU。nano-vllm 的核心路径依赖 CUDA、Triton、flash-attn 和 CUDA Graph，不提供 CPU fallback。

先记录环境：

```bash
nvidia-smi
nvcc --version
python --version
```

在 `reports/environment.md` 中记录：

```text
GPU:
显存:
Compute Capability:
驱动:
CUDA Toolkit:
Python:
PyTorch:
PyTorch CUDA runtime:
Triton:
flash-attn:
模型路径:
Git commit:
```

注意：`nvidia-smi` 显示的驱动支持版本、`nvcc` 的 Toolkit 版本和 PyTorch wheel 自带的 CUDA runtime 不是同一个概念。遇到编译错误时把三者分开检查。

### 1.2 获取源码并创建环境

为了学习源码，不建议只安装 wheel。克隆后使用 editable install：

```bash
git clone https://github.com/GeeeekExplorer/nano-vllm.git
cd nano-vllm
git checkout bb823b3e06983d71485a8e1f23715ebd87d98ef8

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install -e .
```

如果 `flash-attn` 构建失败，按以下顺序定位：

1. `python -c "import torch; print(torch.__version__, torch.version.cuda)"`
2. `nvcc --version`
3. 检查 PyTorch 是否为 CUDA 版本而非 CPU 版本。
4. 检查编译器和 Python 版本是否在依赖支持范围内。
5. 先单独安装与你的 PyTorch/CUDA 组合匹配的 flash-attn；若构建环境看不到已安装的 PyTorch，可尝试 `python -m pip install flash-attn --no-build-isolation`，再执行 editable install。

不要在不知道原因时反复随机更换 CUDA 版本。每次只改变一个依赖并记录。

### 1.3 下载模型

官方 README 给出的命令是：

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

验证配置：

```bash
python - <<'PY'
from transformers import AutoConfig
import os

path = os.path.expanduser("~/huggingface/Qwen3-0.6B")
c = AutoConfig.from_pretrained(path)
for name in [
    "hidden_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "head_dim", "vocab_size", "intermediate_size",
    "max_position_embeddings"
]:
    print(f"{name}: {getattr(c, name, None)}")
print("dtype:", c.dtype)
PY
```

若结果与本文常数不同，以本机模型配置为准，并在所有 kernel 测试中显式传入真实形状。

### 1.4 跑通官方示例

```bash
python example.py
python bench.py
```

第一次只要求：

- 模型成功加载；
- 能完成 prefill 和多步 decode；
- `enforce_eager=True` 能运行；
- `enforce_eager=False` 能完成 CUDA Graph capture 和 replay；
- 记录基线吞吐，不要求与 README 数字一致。

如果 eager 能跑而 Graph 模式失败，暂时不要阻塞 P0–P3。把问题记录下来，P5 专门处理 Graph。

### 1.5 建立自己的基线命令

创建三个固定 workload，后续每个阶段都重复：

| 名称 | 请求 | 目的 |
|---|---|---|
| tiny | 2 条，prompt 16–32，decode 8 | 快速正确性 |
| decode-heavy | 32 条，prompt 128，decode 256 | decode 性能 |
| prefix-mix | 16 条共享 512-token 前缀，suffix 不同 | prefix cache/调度 |

固定 prompt token ids 比固定文本更容易复现。采样阶段应使用固定 exponential noise 或临时增加确定性 argmax reference，不能仅凭“最后文本看起来一样”判断 kernel 正确。

**本仓库的实现：`baseline.py`**（仓库根目录，模型默认本地 `~/huggingface/Qwen3-0.6B`）。三个 workload 均从带固定种子的 `random.Random` 生成固定 token ids（不依赖 tokenizer 版本）；把相邻两次运行（同一 seed / 同一采样模式）的完整输出 token ids 逐 token 比对，逐位一致即确定性复现，并对计时运行的输出 token ids 计算 SHA-256 写入 JSON，供跨阶段比对。

```bash
source .venv/bin/activate

# 每个 workload 默认跑 2 次（第 1 次 warmup/触发编译，第 2 次计时），并报告确定性
python baseline.py tiny                          # 快速正确性：2 条, prompt 16/32, decode 8
python baseline.py decode-heavy                  # decode 性能：32 条, prompt 128, decode 256
python baseline.py prefix-mix                    # prefix cache/调度：16 条共享 512-token 前缀

# 确定性采样两种模式（可叠加 --backend graph / --runs N / --seed S / --print-ids）
python baseline.py tiny --mode argmax            # 确定性 argmax reference（替换采样器为 argmax）
python baseline.py tiny --mode seeded            # 固定 exponential noise（seed torch RNG，默认）
```

对正确性的判断一律看 **`deterministic(两次 token ids 逐位一致)`** 与 `output_ids_sha256`，不是看生成文本。若两次运行不一致，脚本打印首个分歧的 request 索引；注意只有当两次运行的采样随机来源一致时，token ids 相等才有意义（§15.6）。

关键输出解读：

- `prompt_ids_sha256` 唯一标识固定 prompt 内容；换阶段后若报告里的该值不变，说明输入完全一致，可比对输出。
- `prefill Xtok @ ... tok/s | decode Ytok @ ... tok/s`：分阶段吞吐。decode-heavy 的 decode 吞吐是 kernel 性能的稳定参照。
- `prefix-mix` 会额外打印缓存命中证据：主请求输入总 token、实际 prefill 计算量、命中缓存量。实现细节：先单独提交一条「共享前缀 + 4 个 tail token」的预热请求并跑完（前缀块哈希在 scheduler.postprocess 注册），再提交 16 条主请求，使它们只计算各自的 suffix。**tail 不可省略**——若预热请求恰好等于 512 个 token（2 个整块），最后一个块永不查缓存、每次运行都会重算，前缀 KV 产生 bit 级差异，导致两次运行不满足确定性。
- 结果 JSON 写到 `reports/baseline/<workload>_<mode>_<seed>_<backend>.json`，含 `git_commit`、`torch` 版本、KV 块数与 prefix cache 条目数，便于跨阶段比对。

> 采样模式说明：`SamplingParams` 明确禁止 greedy（断言 `temperature > 1e-10`），所以 argmax reference 通过替换 `nanovllm.layers.sampler.Sampler.forward` 实现，必须在 `LLM(...)` 构建前生效。

---

<a id="section-2"></a>

## 2. 开始前的低风险诊断

先独立回答，不查答案。答案在附录 A。这个诊断只用于决定哪里需要补练，不用于评分。

1. 输入形状 `(B, S, 1024)` 经过 Qwen3-0.6B 的 QKV 投影后，Q/K/V 分别是什么形状？为什么 K/V head 少于 Q head？
2. 为什么 prefill 能把很多 token 并行计算，而 decode 每个序列每步只送入一个 token？
3. 600-token 序列、block size 256，需要几个逻辑 block？token 511 位于哪个逻辑 block、offset 是多少？
4. 一个 CUDA kernel 在独立脚本正确，但接入 PyTorch 后偶发读取旧数据。除了 kernel 索引外，你首先检查哪个 stream 问题？
5. 为什么在 kernel 前后直接调用 Python `time.time()` 通常不能正确测量 GPU kernel 时间？
6. 写出稳定 softmax 为什么需要减去最大值，并说明“数值稳定”与“数学结果改变”的区别。
7. 你会怎样证明一个 kernel 是内存带宽受限，而不仅仅是因为 FLOPs 很少？

建议路由：

- 0–2 题能独立回答：先补“CUDA stream/归约/性能测量”微练习，再进入 P0。
- 3–5 题：按标准路线学习。
- 6–7 题：可以压缩 P0/P1 阅读时间，但不能跳过测试与测量基建。

---

<a id="section-3"></a>

## 3. nano-vllm 全局地图

### 3.1 推荐阅读顺序

不要从 Qwen3 模型文件第一行顺序读到最后。按控制流阅读：

1. [`example.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/example.py)
2. [`nanovllm/engine/llm_engine.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/llm_engine.py)
3. [`nanovllm/engine/sequence.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/sequence.py)
4. [`nanovllm/engine/scheduler.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/scheduler.py)
5. [`nanovllm/engine/block_manager.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/block_manager.py)
6. [`nanovllm/engine/model_runner.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py)
7. [`nanovllm/models/qwen3.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/models/qwen3.py)
8. `layers/attention.py → layernorm.py → activation.py → sampler.py`
9. `layers/linear.py → embed_head.py`，最后理解 TP。

### 3.2 一次 generate 的控制流

```text
LLM.generate
  ├─ add_request：tokenize，创建 Sequence，进入 waiting
  └─ while 未完成
       └─ step
            ├─ Scheduler.schedule
            │    ├─ 优先调度 prefill/chunked prefill
            │    └─ 没有 prefill 时调度 decode，必要时 preempt
            ├─ ModelRunner.run
            │    ├─ prepare_prefill 或 prepare_decode
            │    ├─ run_model：eager 或 CUDA Graph replay
            │    └─ Sampler：每个序列产生一个 token
            └─ Scheduler.postprocess
                 ├─ 更新 prefix hash/缓存进度
                 ├─ append token
                 └─ EOS/max_tokens 时释放 blocks
```

核心认知：引擎不是调用一次模型得到整段回答，而是不断执行“调度一个 step → GPU 前向一次 → 每序列增加一个 token”。

### 3.3 Sequence 是请求状态机

重点字段：

| 字段 | 含义 | 谁修改 |
|---|---|---|
| `status` | WAITING/RUNNING/FINISHED | Scheduler |
| `token_ids` | prompt + 已生成 token | `append_token` |
| `num_prompt_tokens` | 固定 prompt 长度 | 初始化 |
| `num_cached_tokens` | 已在 KV cache 中可见的 token 数 | allocate/postprocess/preempt |
| `num_scheduled_tokens` | 当前 step 要计算的 token 数 | Scheduler |
| `is_prefill` | 是否仍处在 prefill/recompute 路径 | Scheduler |
| `block_table` | 逻辑 block 到物理 block id 的映射 | BlockManager |

每次 schedule 前后都检查不变量：

```text
0 <= num_cached_tokens <= num_tokens
0 <= num_scheduled_tokens
num_cached_tokens + num_scheduled_tokens <= num_tokens   # prefill step 内
len(block_table) == ceil(num_tokens / block_size)         # 分配完成后
FINISHED 序列不再占有物理 block
```

### 3.4 Prefill 与 Decode 的数据契约

| 项目 | Prefill | Decode |
|---|---|---|
| 每序列本 step 输入 token | suffix/chunk，长度可变 | 1 |
| 批次展开 | 所有新 token 拼成一维 token batch | shape `(bs,)` |
| 主要元数据 | `cu_seqlens_q/k`, max lengths, slot mapping | context lengths, block tables, slot mapping |
| attention API | `flash_attn_varlen_func` | `flash_attn_with_kvcache` |
| CUDA Graph | 当前实现绕过 | shape 分桶后 capture/replay |
| 主要复用 | 大矩阵计算复用 | 历史 KV，不重算过去 QKV |
| 常见瓶颈 | 依算子/形状而定，GEMM 常更接近计算受限 | 权重/KV 读取和 launch overhead 常重要 |

不能把“prefill 一定计算受限、decode 一定带宽受限”当定律。你要对具体 kernel、batch 和 GPU 做预测并测量。

### 3.5 四种索引必须区分

1. **序列内逻辑位置**：`position`，例如 token 511。
2. **逻辑 block**：`position // block_size`。
3. **物理 block id**：`block_table[logical_block]`。
4. **物理 slot**：`physical_block * block_size + position % block_size`。

KV cache 的最后两维通常是 `[kv_head, head_dim]`。若将其摊平成一行，单层单 token 的 K 或 V 行宽：

```text
D = num_kv_heads × head_dim = 8 × 128 = 1024 elements
```

物理元素地址：

```text
base_element = slot × D
element = base_element + kv_head × head_dim + d
```

---

<a id="section-4"></a>

## 4. 所有阶段共用的正确性与性能纪律

### 4.1 正确性分四层

**层 1：公式正确性**

- 与 PyTorch reference 比较完整输出；
- 同时记录 max absolute error、max relative error、mean error；
- 检查 NaN/Inf；
- 对纯搬运 kernel 争取 bitwise equality。

**层 2：边界正确性**

最低测试集合：

```text
batch:       1, 2, 3, 17, 33
rows:        1, 7, 31, 32, 33, 127
hidden:      128, 1000, 1024, 1536
context:     1, 15, 255, 256, 257, 1023, 1024, 4096
block size:  16/64（合成测试）, 256（真实引擎）
GQA ratio:   1, 2, 4
slot:        正常、跨物理 block、-1
```

真实 `Config` 当前要求 `kvcache_block_size % 256 == 0`，较小 block size 只用于独立合成 kernel 测试，不直接传给引擎。

**层 3：集成正确性**

- reference/custom 开关切换；
- 使用固定输入；
- 比较中间 tensor 或 logits；
- 采样存在随机性时使用固定噪声，不能只比较生成 token。

**层 4：Graph 正确性**

- capture 期间无动态分配、CPU 同步和不支持的 API；
- replay 多次且每次更新输入；
- padded rows 使用 `slot=-1`，不会写坏 cache；
- 不同 graph bucket 都测试。

### 4.2 建议容差

容差不是固定真理，应依据 reference 的累加精度设置：

| 场景 | 初始建议 |
|---|---|
| FP32 elementwise/reduction | `atol=1e-5, rtol=1e-5` |
| BF16/FP16 RMSNorm | `atol=1e-2, rtol=1e-2` |
| FP32 online-softmax state、低精度输入 | 从 `1e-2` 开始，记录长度相关误差 |
| store/gather 纯复制 | bitwise equality |

不要只看 `allclose=True`。上下文长度增加时绘制误差曲线，识别误差是否随序列长度系统增长。

### 4.3 微基准模板

每个测量都要 warmup，并用 CUDA Event：

```python
import torch

def bench_cuda(fn, warmup=50, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iters  # microseconds
```

报告至少包含 median；若逐次测量，再报告 p50/p95。不要在 timed loop 中创建新 tensor、打印或调用 `.item()`。

### 4.4 三种工具的职责

**CUDA Event**：回答一个固定操作平均耗时多少。

**Nsight Systems**：回答 CPU launch gap、同步、H2D copy、kernel 排列和 Graph replay 是否减少提交开销。

```bash
nsys profile -t cuda,nvtx,osrt -o reports/p1_timeline python labs/p1_silu/integration.py
```

**Nsight Compute**：回答单 kernel 的内存吞吐、load/store 效率、occupancy、warp stall 和算术吞吐。

```bash
ncu --set full -o reports/p2_rmsnorm python labs/p2_reduction/bench_rmsnorm.py
```

`--set full` 很慢。初步迭代时只选需要的 section，最终报告再收集完整数据。ncu 可能通过 replay/serialization 改变运行行为，因此不要用它报告端到端延迟。

### 4.5 模式卡模板

每完成一个 kernel，在 `reports/pattern-cards.md` 增加：

```markdown
## 模式名

- 解决的问题：
- 激活信号：什么形状/瓶颈下应考虑它？
- 核心映射：一个 thread/warp/block 负责什么？
- 数据复用位置：register/shared/L2/无复用？
- 同步边界：
- 字节与 FLOPs：
- 常见错误：
- 不适用条件/切换条件：
- 本次实测：
- 改变形状后的迁移结果：
```

---

<a id="section-5"></a>

## 5. P0：建立基线、请求追踪与实验框架（3–5 小时）

### 5.1 阶段目标

拿着一次真实运行的 trace，独立解释每个 step 调度了哪些序列、每个序列计算几个 token、准备了哪些 GPU 元数据，以及 step 后哪些状态发生改变。

先完成这一阶段，后续 kernel 才不是对孤立 tensor 编程。

### 5.2 阅读任务

精读：

- `llm_engine.py:43-55`：add/step
- `llm_engine.py:60-90`：generate 循环
- `sequence.py:14-83`：序列字段与序列化
- `scheduler.py:25-92`：prefill、decode、preempt、postprocess
- `model_runner.py:129-188`：prefill/decode 元数据

第一次阅读只回答：

1. 谁拥有 waiting/running 队列？
2. 谁分配和释放物理 block？
3. 谁知道逻辑 token 对应哪个物理 slot？
4. 谁决定调用 prefill 还是 decode attention？
5. 谁把新 token 加到 Sequence？

### 5.3 动手一：生成三序列 trace

构造三条序列：

- A：长度 20；
- B：长度 300；
- C：长度 600，与 B 共享前 256 token。

暂时设置 `enforce_eager=True`、`max_num_batched_tokens=512`。在自己的调试分支中添加结构化 trace，不要散乱打印 tensor 全值：

```text
step_id, phase              ← 第几步？是 prefill 还是 decode？
seq_id, status              ← 哪些序列参与？它们现在什么状态？
num_tokens, num_cached_tokens, num_scheduled_tokens  ← 每个 seq 的"进度条"
block_table                 ← 每个 seq 的逻辑块→物理块映射
input_ids.shape, positions range  ← 本轮算了哪些 token
slot_mapping.shape/min/max       ← KV 写到了哪些物理槽位
cu_seqlens_q, cu_seqlens_k       ← 前缀和数组
context_lens, block_tables.shape ← decode 阶段的上下文长度
sampled_token                ← 采样出了哪个新 token
```

在运行前先预测前 3 个 step，再与 trace 核对。错误时只修正第一个因果差距，例如“把 num_cached_tokens 误认为 num_prompt_tokens”。

### 5.4 动手二：画状态表

每个 step 后填写：

| step | phase | waiting | running | A cached/total | B cached/total | C cached/total | free blocks |
|---:|---|---|---|---|---|---|---:|

你必须解释：

- Scheduler 为什么一旦调度到 prefill 就立即返回，不与 decode 混合；
- 为什么 decode 的 `num_tokens` 在 `LLMEngine.step` 中用负数表示；
- chunked prefill 为什么只允许当前 prefill batch 的第一个序列被切 chunk；
- 被 preempt 的序列为什么回到 WAITING 并释放 block；
- 当前实现的 recompute preemption 保留 token ids，但丢弃 KV residency。

### 5.5 动手三：建立测试和基准公共函数

在 `labs/common/` 中实现：

- `compare_tensors(ref, out)`：返回最大绝对/相对/平均误差和 NaN 数；
- `bench_cuda(fn)`：统一 warmup 和事件计时；
- `seed_all(seed)`；
- 一组公共 shape cases；
- `assert_no_input_mutation`，除非算子契约明确允许原地修改。

### 5.6 P0 出关标准

- 不看代码，画出 `generate → step → schedule → run → postprocess`。
- 能说明 prefill/decode 的元数据差异。
- Trace 中所有 slot 都能从 block table 手算得到。
- 已建立可复用的 correctness/benchmark 工具。
- 改成 257-token prompt 后，能预测 block 数和末 token slot，再用 trace 验证。

如果做不到，不要进入 kernel 优化；继续缩小到单序列、block size 256 的例子。

---

<a id="section-6"></a>

## 6. P1：第一个自定义 CUDA 算子与引擎接入（6–10 小时）

### 6.1 能力桥接

你已知的 grid-stride elementwise kernel，加上 PyTorch 的当前 stream/device/shape 契约，就成为可插入真实引擎的自定义算子。这里的重点不是 SiLU 公式，而是建立所有后续 kernel 的工程接口。

### 6.2 目标算子

官方 `SiluAndMul.forward`：

```python
x, y = x.chunk(2, -1)
return silu(x) * y
```

输入最后一维为 `2H`，输出最后一维为 `H`：

```text
out[i] = x[i] / (1 + exp(-x[i])) * x[i + H]
```

在 Qwen3 MLP 中，`gate_up_proj` 输出 6144，故 `H=3072`。

### 6.3 先写算子契约

在写代码前写清：

```text
输入：CUDA tensor，FP32/FP16/BF16，最后一维为偶数，当前版本要求 contiguous
输出：同 device、同 dtype，shape 最后一维减半
累加：仅 elementwise，无跨元素累加
允许修改输入：否
Graph：forward 内不得 host sync 或调用裸 `cudaMalloc`；输出由 PyTorch allocator 管理并在 capture/replay 中保持地址契约
尾部：元素总数不要求是 blockDim 或 vector width 的整数倍
```

### 6.4 实现顺序

1. 纯 FP32 scalar kernel。
2. 测试随机二维/三维输入和非对齐元素数。
3. 增加 FP16/BF16 dispatch；在 FP32 中计算激活，再转换输出。
4. 确认 launcher 使用 PyTorch 当前 CUDA stream，而不是硬编码 default stream。
5. 增加 device、dtype、shape、contiguous 检查和 launch error check。
6. 用 reference/custom 开关接入 `SiluAndMul.forward`。
7. eager 集成正确后，再检查 `torch.compile` 和 Graph 交互。

一个合理的 thread 映射：每个线程负责一个输出元素，按 grid-stride 遍历所有 rows × H。初版不要使用 shared memory。

### 6.5 必做测试

```text
shape=(1, 2)
shape=(1, 6144)
shape=(3, 6144)
shape=(17, 6144)
shape=(2, 3, 2000)
含大正数、大负数、0、非有限输入检查
FP32/FP16/BF16
非默认 CUDA stream
```

非默认 stream 测试的目的：在自定义 stream 中先写输入，随后立即启动算子。若 launcher 错用 default stream，可能读到旧值。

### 6.6 性能实验

分别测：

- 空 kernel 的 launch 时间；
- PyTorch/Inductor reference；
- scalar CUDA；
- 适合 dtype 的向量化版本；
- rows `1,2,4,8,16,32,64,128`。

用 nsys 判断 reference 实际产生几个 kernel、是否存在 launch gap。用 ncu 判断自写 kernel 的 load/store 是否合并。不要预设自写版本一定比 Inductor 快。

向量化前必须回答：

- 地址是否满足对齐？
- BF16 应使用什么 packed 类型？不要把 `half2` 当作 BF16 类型。
- 如何处理尾部？
- 向量化是否减少指令，还是仅改变语法？

### 6.7 引擎实验

使用 tiny workload：

1. reference 路径保存每层第一个 MLP 输出摘要；
2. custom 路径使用相同输入比较；
3. 暂时固定采样噪声或比较 logits；
4. 运行完整生成；
5. nsys 确认 custom op 位于预期位置。

### 6.8 P1 出关标准

- 自定义算子在非默认 stream 正确。
- 能说明 PyTorch CUDA extension 为什么必须使用当前 stream/device。
- scalar 与向量版都能处理尾部。
- eager 集成通过，输出误差符合 dtype 预期。
- 能从 nsys 区分 kernel 执行时间与 launch gap。
- 改成合成 `H=1000` 后仍正确，或明确声明只有 `H=3072` 的快速路径并提供通用 fallback。

**模式卡**：`逐元素融合 = 少一次中间 tensor 读写，但收益受 launch、编译器融合和实际 shape 共同决定。`

---

<a id="section-7"></a>

## 7. P2：归约、RMSNorm 与可控 Sampler（8–14 小时）

### 7.1 阶段目标

掌握 block/warp reduction、FP32 累加、同步边界、向量化和带宽分析。最终接入融合 residual RMSNorm，并实现一个随机数来源可控的宽行采样归约。

### 7.2 归约心智模型

典型映射：一行一个 block。

```text
每线程处理 d = tid, tid + blockDim, ...
  ↓
线程局部累加器
  ↓
warp 内 shuffle reduction
  ↓
每 warp 一个 partial 写入 shared memory
  ↓
第一个 warp 合并所有 partial
```

必须能指出：

- block 间没有隐式同步；
- `__shfl_*_sync` 只在 warp 内通信；
- 多 warp block 仍需一次 shared-memory 汇总；
- FP16/BF16 输入不意味着必须使用低精度累加；
- `__syncthreads()` 放错位置会导致数据竞争或死锁。

### 7.3 练习一：独立 row reduction

先实现 `row_sum` 和 `row_max`：

- rows：`1,3,32,127`；
- width：`31,32,33,127,128,129,1000,1024,151936`；
- scalar、shared tree、warp shuffle 三版；
- 与 `torch.sum/max` 比较。

不要直接从 Sampler 开始。row reduction 是更小、错误更可定位的前置能力。

### 7.4 RMSNorm 与 fused residual

公式：

\[
r = x + residual
\]

\[
y_d = w_d \cdot r_d \cdot \operatorname{rsqrt}\left(\frac{1}{H}\sum_j r_j^2 + \epsilon\right)
\]

一个 block 负责一行。一个 kernel 内部可以对行进行两次逻辑遍历：

1. 读取 `x/residual`，FP32 计算 `r` 和平方和；
2. 得到 `inv_rms` 后再次生成 normalized output，并按接口写 residual。

初版不要声称“流量减半”。实际节省多少取决于是否同时输出 residual、Inductor reference 是否已经融合、寄存器能否保留部分数据。应列出 reference 和 custom 的实际全局读写，再用 profiler 验证。

必测：

- `H=128` 的 Q/K norm；
- `H=1024` 的层 RMSNorm；
- 合成 `H=1000/1536`；
- 极小方差、全零、大幅值输入；
- residual 为 None 和 fused residual 两种接口；
- FP32 reference，BF16/FP16 输入。

优化顺序：

1. 正确的 FP32 scalar load；
2. warp shuffle reduction；
3. 多 warp 合并；
4. 合适的 packed load/store；
5. 扫描 blockDim；
6. 比较实际 GB/s 与设备 copy baseline。

### 7.5 Sampler：先把 RNG 从归约中分离

官方实现执行：

```python
logits = logits / temperature
probs = softmax(logits)
token = argmax(probs / E), E ~ Exp(1)
```

因为 softmax 的归一化常数对所有类别相同：

\[
\arg\max_i \frac{\operatorname{softmax}(z/T)_i}{E_i}
= \arg\max_i (z_i/T - \log E_i)
\]

因此教学版本分两步：

1. PyTorch 用固定 seed 生成 `E`；
2. CUDA kernel 逐元素计算 `score=z/T-log(E)`，每行做 argmax。

这样先只学习宽行归约。直接在 kernel 中复刻 PyTorch RNG 会引入 Philox 状态、可复现性和 CUDA Graph capture 等独立问题，放到可选优化。

Sampler 测试：

- 确定性：给定 logits、temperature、E，自写 kernel 必须与 reference token 完全一致；
- 稳定性：E 要 clamp 到正下界，避免 `log(0)`；
- 温度：检查不同 batch row 使用不同 temperature；
- vocabulary 尾部：151936 不假定为 blockDim 整数倍；
- 统计：只对 8–32 个有效类别的受控分布做频数检验，确保每个 bin 有足够期望样本。

### 7.6 带宽判断

RMSNorm 常具有较低算术强度，但“低 FLOPs”不自动等于达到带宽上限。至少提供：

- 输入/输出理论字节；
- kernel 时间；
- effective GB/s；
- copy kernel 的实测 GB/s；
- ncu DRAM throughput、load/store efficiency；
- occupancy 和主要 stall reason。

如果 achieved bandwidth 很低，可能是 launch、访问不合并、占用率、依赖链或 shape 太小，而不是“已经带宽饱和”。

### 7.7 P2 出关标准

- 能从零写出多 warp block reduction，并解释所有同步点。
- RMSNorm 在 `H=128/1000/1024/1536` 上正确。
- custom fused residual 路径接入模型，使用 logits/hidden state 做确定性比较。
- Sampler 在固定 E 下 token 完全一致。
- 能解释为什么该 Sampler 不需要真的计算 softmax。
- 能用测量证据，而非 FLOPs 直觉，说明 kernel 当前受什么限制。

**模式卡**：`按行归约 = 每线程 strided partial + warp 合并 + block 合并；精度、同步和尾部属于正确性契约。`

---

<a id="section-8"></a>

## 8. P3：Paged KV 内存引擎与不规则寻址（12–20 小时）

### 8.1 阶段目标

从“知道 PagedAttention 的名词”提升到能够：

- 手推 block allocate/free/share；
- 证明 block manager 不变量；
- 从 logical token 推导 physical slot；
- 编写 store 与 inverse gather kernel；
- 解释 prefix cache 为什么只共享完整 block。

这一阶段是推理引擎主线的核心，不应只做 GPU kernel。

### 8.2 KV block 容量模型

一个物理 block 跨所有层保存 K 和 V：

\[
B_\text{block}
=2\times n_\text{layers}\times block\_size\times n_\text{kv-heads}\times d_\text{head}\times dtype\_bytes
\]

对 Qwen3-0.6B、BF16：

```text
2 × 28 × 256 × 8 × 128 × 2
= 29,360,128 bytes
≈ 28 MiB
≈ 29.36 MB
```

因此 8GB 显存不等于能全部用于 KV。`allocate_kv_cache` 先测模型 warmup peak，再根据 `gpu_memory_utilization` 计算可用 block 数。你必须读取并解释 `model_runner.py:91-121`。

### 8.3 BlockManager 的关键机制

`BlockManager` 保存：

- `free_block_ids`：可重新分配的物理 block；
- `used_block_ids`：当前至少被一个序列引用；
- `ref_count`：共享 block 的所有者数；
- `hash_to_block_id`：prefix hash 到物理 block；
- 每个 block 的 `hash/token_ids`：用于检测 hash collision 后的真实 token equality。

Prefix hash 是链式的：

```text
h0 = hash(block0, prefix=-1)
h1 = hash(block1, prefix=h0)
h2 = hash(block2, prefix=h1)
```

因此相同的 block1 token，若前缀不同，也不会被视为同一个逻辑 prefix。

当前 `can_allocate` 只检查 `range(seq.num_blocks - 1)`，即最后一个 block 不参与 prefix hit。原因是最后 block 可能未填满，若两个序列共享后继续写入，需要 copy-on-write；当前最小实现没有实现 COW，所以只共享完整、不可再修改的 block。

### 8.4 动手一：BlockManager 性质测试

在 CPU 上先写测试，不需要 GPU：

1. **守恒**：`len(free) + len(used) == num_blocks`。
2. **引用**：每个 used block 的 `ref_count >= 1`；free block 的 `ref_count == 0`。
3. **唯一性**：同一序列的 block table 不应意外重复，除非你明确构造允许的别名场景。
4. **共享**：两个相同完整 prefix 的序列应共享物理 block，ref_count 增加。
5. **释放**：释放一个共享序列不会让另一个序列失去 block。
6. **尾块隔离**：相同但未满的最后 block 不共享。
7. **hash collision 防护**：hash 相同但 token ids 不同不能命中。
8. **抢占恢复**：preempt 后 block table 清空、cached tokens 归零、token ids 保留。

推荐构造 block size 4 的独立 BlockManager 合成测试，便于手算。不要把较小 block size 传给真实 `Config`。

### 8.5 Slot mapping 手算

对 block size 256，若：

```text
seq.block_table = [9, 2, 17]
position = 511
```

则：

```text
logical_block = 511 // 256 = 1
offset = 511 % 256 = 255
physical_block = block_table[1] = 2
slot = 2 × 256 + 255 = 767
```

decode 最新 token 使用：

```text
slot = block_table[-1] × block_size + last_block_num_tokens - 1
```

`-1` 是因为 `last_block_num_tokens` 是计数，slot offset 从 0 开始。

prefill/chunked prefill 更复杂：要从 `start=num_cached_tokens` 到 `end=start+num_scheduled_tokens` 遍历逻辑 blocks，并正确处理首尾部分 block。精读 `model_runner.py:129-169`，手推：

- `start=0,end=300`；
- `start=256,end=600`；
- `start=300,end=600`；
- prefix hit 后 `start=512,end=600`。

### 8.6 动手二：移植 store_kvcache

官方 Triton kernel 的契约：

```text
key/value:     (N, num_kv_heads, head_dim)
k/v cache:     (num_blocks, block_size, num_kv_heads, head_dim)
slot_mapping:  (N,), int32
D:             num_kv_heads × head_dim
```

它把第 `idx` 个新 token 的 K/V 行写入 `slot_mapping[idx]`。当 `slot==-1` 时跳过，这是 CUDA Graph padded row 的安全哨兵。

实现顺序：

1. scalar CUDA copy；
2. `slot=-1`；
3. FP16/BF16；
4. 非连续输入要么拒绝，要么通过 stride 正确访问；
5. 适合 dtype 的 packed copy；
6. 与 Triton bitwise 比较；
7. 接入 `Attention.forward` 的 store 调用点。

不要仅测试连续递增物理 block。至少使用：

```text
slot_mapping = [513, 12, 1024, -1, 255, 256]
```

并在写前给 cache 填充 sentinel，确认未目标位置保持不变。

### 8.7 动手三：inverse gather

输入：paged K/V cache、`block_tables`、`context_lens`。

输出：每个序列物化后的连续 K/V，用于建立 attention reference。初版可以按最大 context padding，另返回有效长度。

对每个 `(seq, pos, kv_head, d)`：

```text
logical_block = pos // block_size
offset = pos % block_size
physical_block = block_tables[seq, logical_block]
out[seq, pos, kv_head, d] = cache[physical_block, offset, kv_head, d]
```

测试必须覆盖：

- block table 非顺序；
- `context_len` 恰为 256；
- 255/257；
- block table 后部为 `-1` padding；
- 两个序列共享一个完整物理 prefix block；
- `gather(store(x)) == x`，但只在有效位置比较。

### 8.8 RoPE CUDA：可选热身

RoPE 是 positions-driven gather 和融合布局转换的好练习，但不是 paged attention 的必要前置，可在时间充足时完成：

- 从 `cos_sin_cache[positions]` gather；
- q heads=16、kv heads=8，不能假定两个 tensor 网格相同；
- FP32 计算旋转后转换回输入 dtype；
- 避免 `chunk + cat` 的中间 tensor；
- 测 positions 重复、乱序和边界值。

### 8.9 P3 出关标准

- 通过所有 BlockManager 性质测试。
- 能手推 prefix hit、partial suffix 和 decode 的 slot mapping。
- CUDA store 与 Triton 对有效位置 bitwise 一致。
- inverse gather 能重建共享/乱序 paged cache。
- 能解释 `slot=-1`、完整 block 共享、ref_count 和 recompute preemption。
- 改变 GQA ratio 或合成 block size 后，索引实现仍正确。

**模式卡**：`Paged KV = 逻辑位置经 block table 间接寻址到物理 slot；scatter 写与 gather 读必须共享同一地址公式。`

---

<a id="section-9"></a>

## 9. P4：Prefill、Decode、Roofline 与调度实验（10–16 小时）

### 9.1 阶段目标

把“prefill 与 decode 不同”从口号变成三个可测模型：

1. 数据形状与元数据模型；
2. FLOPs/bytes/launch 的性能模型；
3. workload 层面的吞吐、TTFT、TPOT/ITL 和 KV 容量模型。

这一阶段不以写复杂 kernel 为主，而是训练推理引擎的系统判断。

### 9.2 Varlen 批次

多个不同长度序列不必 padding 成矩形。将 token 行拼接：

```text
seq0: length 3
seq1: length 5
seq2: length 2
flat tokens: length 10
cu_seqlens = [0, 3, 8, 10]
```

第 `b` 个序列范围：

```text
start = cu_seqlens[b]
end   = cu_seqlens[b + 1]
```

在 prefix-cache prefill 中：

- Q 只包含新 suffix；
- K/V 语义长度包含 cache 中的完整 prefix + 新 suffix；
- 因此 `cu_seqlens_q` 与 `cu_seqlens_k` 可以不同；
- `block_tables` 告诉 flash-attn 去哪里读历史 K/V。

### 9.3 动手一：segmented reduction

实现一个 kernel：输入 flat `(total_tokens, H)` 和 `cu_seqlens`，每个序列输出所有 token/hidden 的和或均值。

学习重点不是求和，而是：

- block 如何映射到 segment；
- segment 长度不均时如何分工；
- 是否出现长 segment 拖尾；
- offset 读取能否保持后续数据连续。

测试：

```text
lengths=[1]
lengths=[3,5,2]
lengths=[0,7,1]       # 若契约允许空 segment；否则明确拒绝
lengths=[1,1024,3]
H=127/128/1024
```

迁移检查：不用 kernel 名提示，给你一组 packed audio frames + offsets，能否识别为同一种 segmented pattern？

### 9.4 Decode 的 KV 流量公式

对单个序列、生成一个 token，跨所有层读取历史 K/V 的理想有效 payload：

\[
B_\text{KV/step}
=n_\text{layers}\times2_{K,V}\times n_\text{kv-heads}
\times d_\text{head}\times L\times dtype\_bytes
\]

Qwen3-0.6B、BF16、`L=4096`：

```text
28 × 2 × 8 × 128 × 4096 × 2
= 469,762,048 B
≈ 448 MiB ≈ 470 MB / sequence / generated token
```

其中：

```text
2 × 28 × 8 × 128 × 2 = 114,688 B
```

是“每增加一个上下文位置，跨全部层增加的 K/V 字节”，不是 L=4096 的总读取量。

这个公式是理想 payload，尚未包含：

- block table/context metadata；
- cache line 和事务浪费；
- 未显式实现 GQA 复用时的重复读取；
- L2 命中；
- Q、输出和其他算子流量。

每层的理想 K/V payload：

\[
2\times8\times128\times4096\times2=16\text{ MiB}
\]

### 9.5 Roofline 基础

算术强度：

\[
AI=\frac{FLOPs}{Bytes\ from\ memory}
\]

硬件 ridge point：

\[
AI_\text{ridge}=\frac{Peak\ FLOP/s}{Peak\ memory\ bandwidth}
\]

若 AI 明显小于 ridge point，理论上更可能受带宽限制；明显大于则更可能受计算限制。但最终必须结合 achieved bandwidth、SM throughput、occupancy 和 launch 情况。

对于 GEMM `A(M,K) × W(K,N)`，粗略：

\[
FLOPs=2MKN
\]

bytes 至少包含 A、W、输出。decode 中 M 很小，权重重用不足；prefill 中 M 大，W 能被更多 token 复用。这个因果关系比“decode GEMM 永远内存受限”更可靠。

### 9.6 动手二：prefill/decode profile

选择至少四组：

| phase | batch | length | 目的 |
|---|---:|---:|---|
| prefill | 1 | 2048 | 长单序列 |
| prefill | 16 | 128–1024 varlen | packed batch |
| decode | 1 | 4096 | 小 batch/长 context |
| decode | 32 | 4096 | 并行度与带宽 |

每组记录：

- end-to-end step time；
- kernel 数和 launch gaps；
- attention、GEMM、RMSNorm 等主要 kernel 时间占比；
- DRAM throughput；
- SM throughput；
- 理论与实测 bytes；
- 你对瓶颈的结论及反证条件。

不要只 profile 第一次运行。编译、warmup、Graph capture 必须排除在稳态测量之外。

### 9.7 动手三：调度与 workload 指标

原始 `bench.py` 只报告总输出吞吐。为学习引擎，扩展记录：

- **TTFT**：请求加入到第一个输出 token 的时间；
- **TPOT**：第一个 token 后，每个输出 token 的平均时间；
- **ITL**：相邻输出 token 的 inter-token latency，报告 p50/p95；
- 总输入/输出吞吐；
- 每 step 的 prefill/decode batch size；
- waiting/running 数量；
- free/used KV blocks；
- preemption 次数；
- prefix hit blocks。

nano-vllm 是 offline API，所有 prompts 在 `generate` 开始时加入。你仍可从每个 Sequence 的调度/完成时间计算这些指标，但不要把它描述为完整在线 serving 的到达模型。

### 9.8 动手四：三组因果实验

#### A. Prefix cache

比较：

- 随机 prompts；
- 共享 256-token prefix；
- 共享 512-token prefix；
- 共享 600-token prefix。

先预测可复用几个完整 block。600-token 前缀只有前两个完整 block 可安全共享，末尾 partial block 不共享。

记录：prefix hit blocks、prefill scheduled tokens、TTFT、总吞吐和 KV block 使用。

#### B. Chunked prefill

扫描：

```text
max_num_batched_tokens = 256, 512, 1024, 4096, 16384
```

workload：一条 4096-token 长 prompt，加 16 条正在 decode 的短请求。当前调度器严格 prefill 优先且不混合 prefill/decode，因此先预测长 prefill 如何影响 decode ITL，再实测。

#### C. Preemption

降低 `gpu_memory_utilization` 或构造足够多长序列，触发 KV block 紧张。记录：

- 哪个序列被从 running 尾部抢占；
- 释放多少 blocks；
- 为什么它需要 recompute；
- 吞吐和延迟付出什么代价。

不要一开始就修改 scheduler。先证明你理解当前策略，再在独立分支做策略实验。

### 9.9 策略修改实验

原路线建议直接允许第二条序列 chunk。更好的学习方式是先写不变量和假设：

```text
假设：允许多个 chunked-prefill 序列可以提升 token budget 利用率。
潜在风险：metadata/attention path 是否支持多个不同 prefix/suffix 组合？
指标：TTFT、ITL p95、吞吐、每 step token budget utilization。
回滚条件：正确性失败，或 tail latency 显著恶化且不符合目标 workload。
```

修改后至少测试：无 prefix、相同 prefix、不同 cached lengths、跨 partial block 四种情况。实验完成后保留 commit，不要在工作目录里手动改回而丢失证据。

### 9.10 P4 出关标准

- 能从 `cu_seqlens` 重建每个序列范围。
- 能推导并正确解释 114,688 B 与约 470 MB 的不同单位含义。
- 对给定 M/N/K、context、batch，先预测可能瓶颈，再用 profile 验证或否定。
- 报告 TTFT、TPOT/ITL、吞吐和 KV blocks，而非只报 tok/s。
- 能说明 prefix cache、chunk size 和 preemption 分别优化什么、牺牲什么。
- 在改变 workload 后重新选择策略，并给出停止/切换条件。

**模式卡**：`性能分类是 shape × hardware × implementation 的结果；Roofline 用来形成可证伪预测，不是给算子贴永久标签。`

---

<a id="section-10"></a>

## 10. P5：CUDA Graph 与连续 KV Online Attention（6–10 小时）

### 10.1 阶段目标

在进入 paged attention 顶点前，分别解决两个前置能力：

1. 什么样的自定义 kernel 可以被 CUDA Graph 安全捕获和重放；
2. 在连续 K/V 上正确实现 streaming/online softmax。

P5 不再强制手写 decode GEMM，也不把 TP/NCCL 塞进单卡主线。

### 10.2 CUDA Graph 心智模型

普通 eager：CPU 每步逐个提交 kernel。Graph：先捕获固定的 GPU 工作拓扑，后续一次 replay 提交整张图。

Graph 主要冻结：

- kernel/operation 顺序；
- tensor 地址；
- shape 和对应 launch configuration；
- capture 中使用的 workspace/address；
- stream 上的依赖关系。

它不意味着输入值固定。nano-vllm 使用静态 `graph_vars`，每次 replay 前把新值 copy 进固定地址。

### 10.3 阅读 nano-vllm Graph 实现

精读 `model_runner.py:195-257`：

- prefill、eager 或 batch>512 时绕过 Graph；
- decode batch 选择第一个 `>=bs` 的 bucket；
- bucket 为 `[1,2,4,8,16,32,...]`；
- input/positions/slot/context/block table copy 到静态 buffer；
- `slot_mapping` 先填 `-1`，padded rows 不写 KV；
- `context_lens` 清零；
- capture 时只捕获 model forward，logits head 和 sampling 在外部执行。

需要注意：`block_tables` 只覆盖本次有效矩形。设计自己的实验时应显式初始化剩余区域，避免把历史 replay 的值误认为当前有效数据。

### 10.4 动手一：最小 Graph 复刻

先捕获三个简单 kernel，例如：

```text
RMSNorm → SiLU-and-mul → elementwise add
```

要求：

- 静态输入/输出地址；
- capture 前 warmup；
- replay 前 copy 新数据；
- 连续 replay 100 次与 eager reference 一致；
- nsys 对比 CPU launch gap 和总时间。

然后将 P1–P3 自定义 kernel 加入 Graph 兼容性矩阵：

| kernel | capture | replay 改输入 | padded row | 多 bucket | 结果 |
|---|---|---|---|---|---|
| SiLU | | | N/A | | |
| RMSNorm | | | | | |
| KV store | | | `slot=-1` | | |

### 10.5 Online softmax 推导

对连续 score 流维护：

- `m`：目前最大值；
- `l`：以 `m` 为基准的指数和；
- `o`：以相同基准累计的加权 V 向量。

处理新 score `s` 和 value `v`：

\[
m' = \max(m,s)
\]

\[
\alpha = \exp(m-m'), \qquad \beta=\exp(s-m')
\]

\[
l' = \alpha l + \beta
\]

\[
o' = \alpha o + \beta v
\]

结束：

\[
output=o/l
\]

若按 tile 处理，新 tile 先得到局部 `m_t,l_t,o_t`，再用同一重缩放关系与全局状态合并。关键不是“在线”这个名字，而是最大值改变后旧累计量必须乘 `exp(m_old-m_new)`。

### 10.6 动手二：连续 KV decode attention

第一版只做：

```text
q: (B, Hq, D)
k/v: (B, L, Hkv, D)  # 连续布局
context_lens: (B,)
D=128
```

实现阶梯：

1. `B=1,Hq=Hkv=1`；
2. 多 q heads、MHA；
3. GQA：`kv_head = q_head // (Hq/Hkv)`；
4. 变长 context；
5. BF16/FP16 输入，FP32 `m/l/o` 累加。

一个易证明 baseline：一个 warp 负责一个 `(seq,q_head)`，每 lane 负责 `D/32=4` 个维度。每个 position：

1. lanes 读取 q/k 分片；
2. warp reduction 得到一个 dot score；
3. 更新共享的 `m/l`；
4. 每 lane 更新自己负责的 4 个输出维度。

这版没有实现同一 GQA group 跨 q-head 的显式 KV 复用。必须诚实记录，而不是仅因模型使用 GQA 就声称流量自动减少。

### 10.7 连续 attention 测试

reference：逐序列物化有效 K/V，使用 FP32 明确计算 `softmax(qk^T*scale)@v`；再与 PyTorch SDPA 对照。

测试：

```text
B=1,3,17
Hq/Hkv=(1/1),(16/16),(16/8),(16/4)
L=1,15,255,256,257,1024,4096
D=64,128（若 kernel 声明支持）
极大/极小 qk score
每个 batch row 不同 context_len
```

绘制 max error 随 L 的曲线。若误差随 L 失控，先检查是否误用 FP16 accumulator，再检查重缩放公式和无效位置 mask。

### 10.8 P5 出关标准

- 能解释 Graph 冻结什么、输入值为什么仍能变化。
- P1–P3 kernel 通过 Graph 兼容矩阵。
- 能不查资料推导 online-softmax 重缩放公式。
- 连续 KV attention 在 GQA、变长和 L=4096 上正确。
- 明确说明 baseline 的 KV 重复读取和并行度边界。

**模式卡**：`Online softmax = 最大值变化时同时重缩放旧分母和旧输出累计；它解决单遍流式数值稳定，不自动解决并行度与数据复用。`

---

<a id="section-11"></a>

## 11. P6：顶点项目——Paged Decode Attention（15–30 小时）

### 11.1 目标与边界

替换 `Attention.forward` decode 分支中的 `flash_attn_with_kvcache`。保留以下成熟库边界：

- GEMM 继续使用 PyTorch/cuBLAS；
- prefill 继续使用 `flash_attn_varlen_func`；
- 多卡 collective 继续使用 NCCL/PyTorch distributed；
- 本阶段只重写 decode attention。

成功标准分层：

1. **必过**：正确、边界完整、可接入、可 Graph replay；
2. **必过**：能够用 profiler 解释性能；
3. **优化目标**：基于证据提高 bandwidth/occupancy；
4. **荣誉目标**：在指定 shape 上接近成熟 flash-attn 路径。不得把“2×以内”作为普适及格线。

### 11.2 输入契约

```text
q:             (bs, num_q_heads, head_dim)
k_cache:       (num_blocks, block_size, num_kv_heads, head_dim)
v_cache:       同上
block_tables:  (bs, max_num_blocks), int32，尾部 -1
context_lens:  (bs,), int32
scale:         head_dim ** -0.5
output:        (bs, num_q_heads, head_dim)
```

当前执行顺序是：本 step 新 K/V 先由 `store_kvcache` 写入 slot，然后 decode attention 用 `context_lens` 读取包含该 token 在内的 cache。

### 11.3 地址公式

对序列 `b`、上下文位置 `pos`：

```text
logical_block  = pos // block_size
offset         = pos % block_size
physical_block = block_tables[b, logical_block]
K = k_cache[physical_block, offset, kv_head, :]
V = v_cache[physical_block, offset, kv_head, :]
```

循环只到 `context_lens[b]-1`。如果有效范围内出现 `physical_block==-1`，这是上游 metadata 错误，debug 版本应报告；`-1` 正常只出现在 block table padding 或 Graph padded row。

### 11.4 Version A：分页寻址正确性 baseline

把 P5 连续 attention 的 K/V 读取替换为 block-table 间接寻址，其余映射不变：一个 warp／`(seq,q_head)`。

先不要做：

- shared-memory GQA 复用；
- split-KV；
- persistent kernel；
- 低精度 softmax state；
- 复杂预取流水。

Version A 的任务是把数值问题与分页地址问题分离。调试时同时用两种 reference：

1. P3 inverse gather 物化连续 K/V，再用 P5 kernel；
2. `flash_attn_with_kvcache`。

若 1 正确而 paged 版错误，问题在地址；若 P5 连续版已经错，不能归咎于分页。

### 11.5 Version A 必测矩阵

```text
bs:              1, 3, 16, 33
context:         1, 255, 256, 257, 1024, 4096
q/kv heads:      16/16, 16/8, 16/4
physical blocks: 连续、逆序、随机置换、两序列共享 prefix
padding:         block table 尾部 -1
Graph padding:   bucket bs > actual bs，slot=-1/context_len=0
dtype:           BF16/FP16 输入，FP32 online state
```

对每个 case 记录：最大误差、首个错误 index、对应的 `(b,q_head,pos,physical_block,offset)`。不要只打印整块 tensor。

### 11.6 理论流量要分“按 KV head”与“按 Q head”

理想地，每层每序列只读取一次所有 KV heads：

\[
B_\text{ideal}=2_{K,V}\times L\times n_\text{kv-heads}\times D\times dtype\_bytes
\]

对 BF16、`L=4096,n_kv_heads=8,D=128`，每层为 16 MiB。

但一个 warp／q-head 的 Version A 会让共享同一 KV head 的多个 q heads 各自读取 K/V。若 GQA ratio=2，理论请求流量可能接近理想 payload 的 2 倍，实际 DRAM 流量取决于 L1/L2/cache line 复用。

因此报告中同时列出：

- 理想 unique KV payload；
- 按 kernel 映射计算的请求字节；
- ncu 实测 DRAM bytes；
- 两者差异的解释。

### 11.7 Version B：显式 GQA 复用

若 Version A 正确，尝试一个 CTA 负责 `(seq,kv_head)`：

- CTA 内 `group_size = num_q_heads/num_kv_heads` 个 warp 分别负责 q heads；
- 按 context tile 将 K/V 加载到 shared memory；
- 同一 tile 被组内 q-head warps 复用；
- 每个 q head 维护独立 `m/l/o`；
- tile 之间使用 online-softmax 合并。

设计前计算 shared memory：

```text
tile_tokens × 2(K,V) × head_dim × dtype_bytes
```

例如 BF16、head_dim=128、tile_tokens=16：

```text
16 × 2 × 128 × 2 = 8192 bytes
```

还需考虑 padding、q 存储和 bank conflict。不要默认 shared memory 越大越好；更大 tile 可能降低 occupancy。

Version B 的激活条件：

- GQA ratio > 1；
- Version A 实测存在重复 DRAM 读取或带宽瓶颈；
- shared-memory 占用不会让 occupancy/并行度恶化超过复用收益。

若实测 L2 已很好地服务重复读取，Version B 不一定更快。保留 Version A 作为可切换 baseline。

### 11.8 Version C：长上下文并行度，可选 split-KV

一个 warp/CTA 串行遍历 4096 token 可能暴露不足的并行度或长依赖链。可选 split-KV：

1. 把 context 分成多个 split；
2. 每个 split 独立产生局部 `m_s,l_s,o_s`；
3. 第二阶段 kernel 合并 splits；
4. 合并时仍需按全局 max 重缩放。

激活条件：长 context、batch/head 并行度不足、profile 显示 memory latency/依赖链限制。切换条件：split 中间写入和第二次 launch 的成本超过并行收益。

### 11.9 接入真实模型

在 `Attention` 中增加显式 backend 开关，例如概念上：

```text
decode_backend = "flash" | "cuda_baseline" | "cuda_gqa"
```

不要用注释/取消注释切换。接入检查：

1. tiny workload eager；
2. 比较每层 attention 输出；
3. 比较最终 logits；
4. 固定采样噪声比较 token；
5. prefix hit workload；
6. block 跨界；
7. Graph capture/replay；
8. graph bucket `1,3→4,17→32`。

“逐 token 一致”只有在采样随机源一致时才有意义。否则以 attention output/logits 的数值比较为主要证据。

### 11.10 性能报告

对 `bs=1,16,32,64`、`L=256,1024,4096`：

| backend | bs | L | time us | ideal bytes | DRAM bytes | GB/s | occupancy | max error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flash | | | | | | | | |
| baseline | | | | | | | | |
| GQA reuse | | | | | | | | |

分析顺序：

1. 输出是否正确；
2. kernel 是否足够并行；
3. load 是否合并；
4. DRAM throughput 是否接近设备可达值；
5. register/shared memory 是否限制 occupancy；
6. block table 间接寻址开销多大；
7. GQA 复用是否真的降低 DRAM bytes；
8. split-KV 的额外 launch/intermediate 是否回本。

### 11.11 P6 出关标准

- Version A 覆盖所有分页/GQA/边界 case 并正确。
- 能推导 online-softmax，解释 FP32 state 的必要性。
- 能指出 Version A 没有显式 GQA 复用，不能混淆模型语义与 kernel 数据复用。
- 接入真实 decode 且 eager/Graph 都正确。
- 性能报告包含理论与实测 bytes，而不仅是 wall time。
- 能说明成熟 flash kernel 还做了哪些工作：tiling、并行度、流水、split-KV、架构特化等。
- 换成 GQA ratio=4 或随机 block table 后仍能独立验证。

**模式卡**：`Paged decode attention = online-softmax 状态机 × block-table 间接寻址 × head/warp 映射；正确性 baseline 与复用优化必须分层。`

---

<a id="section-12"></a>

## 12. 可选分支

### 12.1 分支 A：Decode GEMV/GEMM

这是一条高价值 CUDA 性能支线，但不是 paged attention 的前置。

选择两种真实形状：

```text
(bs,1024) @ (4096,1024)^T
(bs,3072) @ (1024,3072)^T
```

学习目标：

- 小 M 下权重读取占主导的原因；
- 一个 block/warp 负责一个输出元素或 tile 的取舍；
- batch 增加如何提高 W 复用；
- 与 PyTorch/cuBLAS 的差距来自 tiling、流水、Tensor Core 和调度。

及格线是解释差距并完成一次有效优化，不要求打败 cuBLAS。若希望深入，转向 CUTLASS/CuTe，而不是无限堆叠自写朴素 GEMV trick。

### 12.2 分支 B：Tensor Parallelism 与 NCCL

先读：

- `linear.py:54-128`：Column/Merged/QKV shard；
- `linear.py:131-156`：RowParallelLinear 与 all-reduce；
- `embed_head.py:9-42`：vocab embedding mask + all-reduce；
- `embed_head.py:45-66`：LM head gather；
- `model_runner.py:15-89`：多进程、NCCL、共享内存控制。

必须能解释：

- ColumnParallelLinear 切 output features，局部输出可以直接交给相应后续 shard；
- RowParallelLinear 切 input features，每张卡只产生部分和，必须 all-reduce；
- bias 只在 rank 0 的局部 F.linear 中加入，all-reduce 后恰好出现一次；
- Q/K/V head 数不同，使 QKV packed weight loader 的 shard offset 更复杂；
- decode 每步对小 tensor 做 collective，通信/launch 难以摊薄。

2+ GPU 时：比较 TP=1/2 的 prefill/decode 吞吐、all-reduce 时间和 scaling efficiency。单卡时只做 tensor shard 手推和小型模拟，不必写“假 all-reduce kernel”冒充真实通信实验。

### 12.3 分支 C：Varlen Prefill FlashAttention

这是 P6 的超集：

- 多 query token；
- causal mask；
- `BLOCK_M × BLOCK_N` 二维 tiling；
- 不同序列长度和 `cu_seqlens`；
- tile 间 streaming-softmax；
- 更复杂的 shared memory/register 调度。

实现阶梯：固定长度 contiguous → causal → varlen → prefix block table。以正确和解释成熟实现差距为主，达到库的 2–3 倍以内属于挑战目标。

### 12.4 分支 D：RNG 与 Graph-safe Sampler

如果希望真正融合 Sampler：

- 理解 PyTorch CUDA RNG/Philox counter；
- 每个元素如何获得不重叠随机数；
- seed/offset 如何随 replay 前进；
- Graph capture 中 RNG state 的约束；
- 统计分布与确定性 replay 如何分别测试。

在完成固定噪声 sampler 之前不要进入此分支。

### 12.5 Persistent kernel

当前主线没有以 persistent kernel 为前置。只有当 profile 证明重复 launch、常驻权重/状态和工作队列能带来明确收益时再学习。不要因为它“高级”就把它塞进课程。

---

<a id="section-13"></a>

## 13. 最终综合项目

### 13.1 Workload

至少构造：

1. 8 条共享 512-token prefix、不同 64-token suffix、decode 128；
2. 8 条无共享 prefix、prompt 128–2048、decode 128；
3. 一条 4096-token 长 prompt 与 31 条短 decode 请求；
4. KV 容量受压并触发 preemption 的 workload。

### 13.2 先提交预测

运行前写下：

- 每类请求可命中几个 prefix blocks；
- 预期 prefill/decode step 序列；
- 峰值 block 使用；
- 哪类请求 TTFT/ITL 最差；
- Flash/baseline/GQA-reuse 三个 attention backend 的相对表现；
- 哪个条件下你会切换 backend 或 scheduler 参数。

### 13.3 最终报告结构

```markdown
# 实验环境与固定提交
# 引擎数据流和关键不变量
# Workload 与预实验预测
# 正确性方法和边界矩阵
# CUDA kernel 映射与数值策略
# 理论 FLOPs/bytes
# nsys 时间线
# ncu counters
# Eager vs Graph
# Prefix/chunk/preemption 的指标
# Reference vs custom 性能表
# 失败尝试与因果诊断
# 适用边界与切换条件
# 改变形状后的迁移测试
# 下一步优化
```

### 13.4 最终通过标准

只有同时满足才算完成：

- 能独立修改引擎，不依赖散乱打印猜测；
- block manager 和 slot mapping 有性质测试；
- 自定义算子遵守 PyTorch stream/device/Graph 契约；
- paged attention 在改变后的 shape 上正确；
- 性能结论由时间线、counters 和理论模型共同支持；
- 清楚说明没实现什么，以及何时应回到 cuBLAS/flash-attn/NCCL；
- 一周后不看笔记，仍能重画控制流、地址公式和 online-softmax 更新。

---

<a id="section-14"></a>

## 14. 推荐周计划

### 14.1 每周 8–10 小时，约 12–14 周

| 周 | 内容 | 当周交付 |
|---:|---|---|
| 1 | 环境、P0 trace、测试基建 | environment + 三序列 trace |
| 2 | P1 scalar SiLU、extension 接口 | correctness matrix |
| 3 | P1 向量化、接入、nsys/ncu | P1 pattern card |
| 4 | P2 row reduction | sum/max 三版 |
| 5 | P2 RMSNorm | fused norm report |
| 6 | P2 Sampler | 固定噪声 + 统计实验 |
| 7 | P3 BlockManager | 性质测试 + 手算表 |
| 8 | P3 store/gather | bitwise/store-gather 测试 |
| 9 | P4 varlen/roofline | prefill/decode profile |
| 10 | P4 scheduler workload | TTFT/ITL/KV 报告 |
| 11 | P5 Graph + contiguous attention | Graph 矩阵 + online attention |
| 12 | P6 paged Version A | 完整边界正确性 |
| 13 | P6 GQA/性能 | backend 对比表 |
| 14 | 综合 workload 与复盘 | final-report.md |

### 14.2 每周 15–20 小时，约 7–9 周

可合并 P0/P1、P2 reduction/RMSNorm、P3 CPU/GPU 两半，但不要合并以下调试边界：

- 连续 attention 与 paged attention；
- paged correctness baseline 与 GQA shared-memory 优化；
- eager 集成与 Graph replay；
- 单 kernel counters 与端到端 workload 指标。

### 14.3 每日节奏

推荐单次 90–150 分钟：

```text
15 min  无笔记回忆上一节地址/公式
20 min  写预测和最小测试
45–75 min 实现/调试
20 min  测量并记录
10 min  写模式卡和下一步
```

如果连续两次 session 都在修同一个不可定位错误，停止性能优化，缩小到合成输入和第一处错误 index。

---

<a id="section-15"></a>

## 15. 高频错误与排查顺序

### 15.1 结果完全错误

1. shape/stride 契约；
2. dtype dispatch；
3. 当前 device/stream；
4. logical block/physical block/slot 混淆；
5. off-by-one；
6. 未初始化输出或 padded region；
7. kernel launch error。

### 15.2 只在 256/257 边界错误

重点检查：

- `pos // block_size` 与 `%`；
- end 是闭区间还是开区间；
- `last_block_num_tokens - 1`；
- prefill 的 start/end block；
- partial first/last block。

### 15.3 短序列正确，长序列出现 NaN/偏差

重点检查：

- softmax max subtraction；
- online-softmax 旧状态重缩放；
- `m/l/o` 是否 FP32；
- exp 输入范围；
- 无效 positions 是否参与；
- reduction 是否漏 lane。

### 15.4 eager 正确，Graph 错误

重点检查：

- capture 后地址是否改变；
- capture 内动态分配/同步；
- padded row 的 `slot=-1`；
- context length 是否清零；
- block table 剩余区域是否有陈旧值；
- kernel launch shape 是否依赖 host-side 动态值；
- RNG state。

### 15.5 kernel 正确但变慢

按顺序问：

1. 比较对象是否已经被 Inductor/Triton 融合？
2. timed region 是否包含分配/编译？
3. shape 是否小到 launch 主导？
4. global accesses 是否合并？
5. vectorization 是否增加寄存器或降低 occupancy？
6. shared memory 是否真的带来复用？
7. 是否为了减少 bytes 增加了更多 launch/intermediate？
8. 成熟库是否使用 Tensor Core、split-KV 或架构特化？

### 15.6 最终 token 不一致

先不要判定模型错：

- Sampler 是否使用同一随机噪声？
- 微小 logits 差异是否改变了随机选择？
- 比较 attention output、hidden state、logits 的第一处偏差；
- 使用受控 argmax/固定 E 重试。

---

<a id="section-16"></a>

## 16. 关键公式速查

### 16.1 逻辑位置到 KV 地址

```text
logical_block  = pos // block_size
offset         = pos % block_size
physical_block = block_table[logical_block]
slot           = physical_block * block_size + offset
element        = slot * (num_kv_heads * head_dim) + kv_head * head_dim + d
```

### 16.2 KV block bytes

\[
2\times layers\times block\_size\times kv\_heads\times head\_dim\times dtype\_bytes
\]

### 16.3 Decode KV payload

\[
layers\times2\times kv\_heads\times head\_dim\times context\_len\times dtype\_bytes
\]

### 16.4 GEMM

\[
FLOPs=2MKN
\]

### 16.5 RMSNorm

\[
y_i=w_i x_i/\sqrt{\frac{1}{H}\sum_jx_j^2+\epsilon}
\]

### 16.6 Online softmax

```text
m_new = max(m_old, s)
alpha = exp(m_old - m_new)
beta  = exp(s - m_new)
l_new = alpha * l_old + beta
o_new = alpha * o_old + beta * v
result = o / l
```

### 16.7 Gumbel/Exponential sampling

```text
argmax(softmax(logits/T) / E)
= argmax(logits/T - log(E)), E ~ Exp(1)
```

### 16.8 Ring all-reduce 通信量模型

对 n 个 rank、消息大小 S，每 rank 的典型发送/接收数据量近似：

\[
2\frac{n-1}{n}S
\]

实际时间还包含 collective latency、拓扑、链路和实现流水。

---

<a id="section-17"></a>

## 17. 自测题

### 17.1 引擎与调度

1. 走一遍 `add_request → schedule → run → postprocess`，列出每步修改的 Sequence 字段。
2. 为什么当前 scheduler 返回纯 prefill 或纯 decode batch？这与更完整生产引擎的混合调度有什么边界？
3. `preempt()` 保留什么、丢弃什么？恢复成本是什么？
4. 为什么 chunked prefill 的 token budget 会影响 TTFT 和 decode ITL？
5. 为什么总 tok/s 无法完整评价 serving 体验？

### 17.2 Paged KV

6. 600-token prompt、block size 256，一共有几个 block，prefix matcher 最多检查几个？
7. 为什么未满 block 不能在没有 copy-on-write 时安全共享？
8. 给定 `block_table=[9,2,17]`，求 positions 0、255、256、511、512 的物理 slot。
9. 为什么 prefix-cache prefill 中 `cu_seqlens_q != cu_seqlens_k`？
10. `slot=-1` 在 Graph replay 中防止了什么？

### 17.3 CUDA 与性能

11. 一个 warp/q-head 的 GQA kernel 是否自动只读一次共享 KV？为什么？
12. 何时应使用 shared memory，何时 L2 已足够？你需要什么测量？
13. nsys、ncu、CUDA Event 分别回答什么问题？
14. 为什么 FP16/BF16 输入仍应使用 FP32 softmax state？
15. 推导 L=4096 时约 470MB 的 KV payload，并解释它不等于实际 DRAM bytes。
16. 为什么 decode 小 GEMM 权重复用通常差于 prefill？
17. Graph capture 为什么需要静态地址而非静态数值？

### 17.4 迁移与边界

18. 若 q heads=32、kv heads=4，P6 Version B 的 CTA/warp 映射如何改变？
19. 若 block size 从 256 改为 64，哪些公式不变，哪些 metadata/测试要改变？
20. 若 profile 显示 Version A 的重复 GQA 读取几乎全部命中 L2，是否仍应做 shared-memory 版？给出停止条件。
21. 若上下文从 4096 增到 32768，但 batch 降到 1，为什么 split-KV 可能更重要？

答案要点见附录 B。先写出自己的推理，再核对。

---

<a id="section-18"></a>

## 18. 推荐资料与阅读边界

### 18.1 直接相关源码

- [nano-vllm 固定提交](https://github.com/GeeeekExplorer/nano-vllm/tree/bb823b3e06983d71485a8e1f23715ebd87d98ef8)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)
- [FlashAttention-2 论文](https://arxiv.org/abs/2307.08691)

### 18.2 CUDA

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- *Programming Massively Parallel Processors*：重点看 memory、reduction、tiling。

### 18.3 按阶段读取，不要预读全部

| 阶段 | 最小阅读 |
|---|---|
| P1 | CUDA launch/stream；PyTorch C++/CUDA extension |
| P2 | reduction、warp shuffle、memory bandwidth |
| P3 | PagedAttention 论文的内存管理部分；Triton store kernel |
| P4 | Roofline 与 Nsight 工具说明 |
| P5 | CUDA Graph；online softmax/FA2 Sec.3.2 |
| P6 | Flash-Decoding/split-KV；成熟 attention kernel 的映射 |
| TP 分支 | Megatron tensor parallel；NCCL collective |

阅读资料的停止条件：当你已经能完成当前阶段的预测、实现和边界判断时，立即回到代码。不要以“读完所有论文”代替实践。

---

## 附录 A：开始前诊断答案

1. Q 为 `(B,S,16,128)`，K/V 为 `(B,S,8,128)`；GQA 让多个 query heads 共享同一 KV head，减少 KV cache 容量，但是否减少实际读取还取决于 kernel 是否实现数据复用。
2. Prefill 的所有 prompt token 已知，可以因果并行；decode 的下一个 token 依赖刚生成的 token，每序列每步只能推进一个位置。
3. `ceil(600/256)=3`；position 511 位于 logical block 1、offset 255。
4. 检查 launcher 是否使用 PyTorch current stream，以及输入生产者和消费者是否在正确 stream 上建立依赖。
5. CUDA launch 对 CPU 异步；Python 计时只测到提交，除非正确同步。微基准应使用 CUDA Event。
6. 减最大值利用 softmax 对公共平移不变，避免 exp 溢出；数学概率不变，只改变数值计算路径。
7. 计算理论 bytes/AI，测 effective/DRAM bandwidth，与 copy baseline 和硬件上限比较，同时排除 launch、低 occupancy、非合并访问等原因。

---

## 附录 B：综合自测答案要点

1. add 创建 Sequence；schedule 分配 block、设置 scheduled/status/is_prefill；run 准备 metadata 并产生 token；postprocess 更新 hash/cached、append、finish/deallocate。
2. 当前实现 prefill 优先且发现 prefill batch 后立即返回，简化 metadata/attention path；不能把它等同于所有生产 vLLM 版本的调度能力。
3. 保留 token ids、采样配置和请求身份；丢弃 block table/KV residency，恢复时重算。
4. chunk 越小，长 prefill 占用单 step 的时间可能下降，但 step/launch/调度开销增多；当前纯 phase 调度下仍可能阻塞 decode。
5. tok/s 不显示首 token 等待和相邻 token 抖动。
6. 3 个 block，最多匹配前 2 个完整 block。
7. 两序列之后会在同一 partial block 追加不同 token；无 COW 会互相覆盖。
8. slots 分别为 `9*256+0`、`9*256+255`、`2*256+0`、`2*256+255`、`17*256+0`。
9. Q 只含新 suffix，K/V 语义上含 prefix+suffix，历史部分经 block table 从 cache 读取。
10. Graph bucket 中多余 rows 不应执行有效 KV store；`-1` 让 store kernel 跳过。
11. 不会。两个 q-head warps 会发出独立 load，除非 CTA/shared-memory 显式复用或 cache 合并请求。
12. 当 producer/consumer 能在 CTA 内复用且 global/cache 流量是瓶颈时考虑 shared；用 DRAM bytes、L2 hit、shared 占用和最终时间验证。
13. Event 测 GPU 时间；nsys 看时间线/launch/sync；ncu 看单 kernel 微架构 counters。
14. 长序列指数和与输出累计对精度敏感，低精度容易累计误差、下溢或状态失真。
15. 见 9.4；实际 DRAM bytes 还取决于重复读取、cache、事务和实现。
16. decode 的 M 小，同一份 W 服务的行少；prefill M 大，可在 tile/batch 中摊薄权重读取。
17. replay 需要节点读写同一批地址和 workspace，但 replay 前可以把新值写入这些静态 buffer。
18. GQA group size 为 8；一个 CTA/kv-head 若一 warp/q-head 需要 8 warps，可能超过理想 CTA 设计，应分组、tile 或重新映射并测 occupancy。
19. `logical_block/offset/slot` 公式不变；block table 长度、容量、边界 case、allocator 预算和 metadata shape 改变。
20. 不一定。若 shared 版提高资源占用且 DRAM bytes 没有显著下降，应保留 Version A；停止条件由时间与 counters 共同决定。
21. 只有少量 head/batch work items 串行扫描超长 context，并行度和依赖链不足；拆分 context 可增加并行 work，但要支付中间状态和 merge 成本。

---

## 附录 C：实验记录模板

```markdown
# 实验名称

## 问题
我想判断什么？

## 输入与环境
- commit:
- GPU/dtype:
- shapes:
- backend:

## 预测
- thread/warp/block 映射：
- 理论 bytes：
- 理论 FLOPs：
- 预计瓶颈：
- 可能失败边界：
- 停止/切换条件：

## 正确性
- reference:
- cases:
- max abs/rel error:
- 首个失败 index:

## 性能
- warmup/iters:
- p50/p95:
- nsys 观察:
- ncu 观察:
- effective GB/s:

## 结论
- 预测是否成立：
- 主要因果差距：
- 下一次只改变什么：

## 迁移
- 改变的 shape/workload:
- 结果:
```

---

## 附录 D：完成后的能力清单

你应当能够不依赖本文完成：

- [ ] 画出请求状态机与每 step 控制流。
- [ ] 手推任意 token 的 paged KV 地址。
- [ ] 解释 prefix hash、full-block sharing 和 ref_count。
- [ ] 写 graph-safe PyTorch CUDA extension。
- [ ] 写正确的多 warp reduction。
- [ ] 解释并测量融合算子的真实流量收益。
- [ ] 推导 prefill/decode 的 shape 与性能差异。
- [ ] 使用 Event/nsys/ncu 回答不同层级的问题。
- [ ] 推导 online softmax。
- [ ] 写连续和 paged decode attention baseline。
- [ ] 区分 GQA 模型语义与 kernel 显式 KV 复用。
- [ ] 让自定义 decode 路径在 CUDA Graph 中工作。
- [ ] 用 TTFT/ITL/吞吐/KV 容量共同评价调度策略。
- [ ] 说明 cuBLAS、flash-attn、NCCL 的保留边界。
- [ ] 在改变后的模型形状/workload 中重新做出选择。

完成这份清单时，你掌握的不只是 nano-vllm 的实现，而是一组可以迁移到 vLLM、SGLang、TensorRT-LLM 或自研推理引擎的核心问题分解方法。
