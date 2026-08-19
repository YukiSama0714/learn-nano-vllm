# nano-vLLM 项目问题、实现与事实审计

> 审计日期：2026-08-19<br>
> 改造基线：`origin/main` / `457e3c7`<br>
> 被审实现：`codex/slo-aware-scheduler` / `2955070`<br>
> 证据范围：本项目完整对话、仓库源码、提交历史、测试代码和对话中贴出的
> RTX 5090 实验结果。本文不使用事后猜测替代缺失日志。

## 0. 为什么需要这份文档

项目经历了环境安装、GPU 容器故障、调度器迭代、kernel 正反实验、指标修正、
PagedAttention 和 CUDA Graph 等多条并行路线。旧文档有一些状态已经过期：

- 把已经实现但尚未在服务器验证的 CUDA Graph 写成“未来工作”；
- 把已经完成且端到端失败的 split-K 写成“待实现/待验证”；
- 把“代码存在”“单测通过”“GPU correctness 通过”“端到端达标”混为一谈；
- 项目 diff 统计仍停留在旧提交。

本文的目的不是包装项目，而是建立一份面试时经得起追问的事实账本。

## 1. 状态标签：以后统一按这一套说法

| 标签 | 含义 | 可以对外怎么说 |
|---|---|---|
| `CODE` | 源码路径已实现 | “实现了该路径” |
| `UNIT` | CPU/逻辑单元测试通过 | “状态机/边界条件有单测” |
| `GPU-CORRECT` | 在 5090 上与 reference 通过数值容差 | “kernel 正确性通过指定矩阵” |
| `MICRO` | 独立算子微基准有数据 | “局部 kernel 在该形状更快/更慢” |
| `E2E` | 固定 workload 的请求级 A/B 有数据 | “端到端在该 workload 的实际结果” |
| `ACCEPTED` | 达到预先写明的验收阈值 | “达到项目目标” |
| `UNVERIFIED` | 代码存在，但缺少对应层级验证 | 只能说“实验性实现” |
| `REJECTED` | 正确但端到端负收益或未达门槛 | “保留为负实验，不默认启用” |

特别注意：`CODE != GPU-CORRECT != E2E != ACCEPTED`。

## 2. 审计结论先行

相对 `origin/main`，截至 `2955070` 的实际改动为：

```text
61 files changed
9521 insertions
351 deletions
```

### 2.1 当前最可信的成果

`slo_aware_v2` 是证据最完整的核心成果。在 Qwen3-8B、RTX 5090、held-out
320 请求实验中：

| 指标 | prefill-first | slo-aware v2 | 变化 |
|---|---:|---:|---:|
| Max ITL P95 | 259.21ms | 68.23ms | -73.7% |
| 请求发生 Max ITL > 75ms | 93.8% | 0.6% | -93.2pp |
| Output tok/s | 255.56 | 255.16 | -0.16% |
| TTFT P95 | 156.27ms | 213.56ms | +36.7% |
| E2E P95 | 2180.29ms | 2353.69ms | +8.0% |

这不是“所有延迟都更小”，而是用几乎不变的吞吐和仍满足 500ms 的 TTFT，
换取显著更平滑的流式输出。

### 2.2 当前不能宣称完成的部分

- `slo_aware_v3` 已实现 mixed batch，但没有达到路线中规定的最终性能验收；
- Triton PagedAttention general kernel 正确，但 page32 eager E2E 只有 Flash
  eager 吞吐的 84.1%；
- split-K kernel 的 GPU micro 有收益，但 eager E2E 吞吐下降 8.4%，属于
  `REJECTED`；
- Triton PagedAttention CUDA Graph 已写入源码，但服务器 smoke 尚未给出成功
  结果，且 code review 发现 auto 路由语义偏差；
- n-gram 和 0.6B draft speculative 路径存在并有单测，但没有完成 100 条固定
  greedy、acceptance、TPOT、32GB 显存和 10% 改善验收；
- 没有完成 HTTP/OpenAI 服务、多卡优化、FP8 KV、量化、LoRA 或异步流水线。

### 2.3 最初的 batch scaling 只是一条启动自检

环境刚搭好时，8B 固定 512 输入/128 输出的同步 batch smoke 得到：

| Batch | Output tok/s | Total tok/s | Peak allocated GiB |
|---:|---:|---:|---:|
| 1 | 92.37 | 461.84 | 27.23 |
| 4 | 229.49 | 1147.46 | 27.40 |
| 8 | 588.76 | 2943.80 | 27.61 |

它证明模型能运行且 continuous batching 有效，但 workload、arrival 和后来的
Poisson SLO 实验不同，不能拿这些数字与 v2/v3 表格直接比较。

## 3. 最终稳定的软件与硬件环境

对话中出现过多个中间版本；正式实验应以最后确认的环境为准：

```text
GPU                 NVIDIA GeForce RTX 5090 32GB
Compute capability  12.0
PyTorch             2.8.0+cu128
Triton              3.4.0
CUDA runtime        12.8
FlashAttention      2.8.3, cu12, torch2.8, CXX11 ABI TRUE, cp312
Python              3.12 virtualenv
Target model        Qwen3-8B
Draft model         Qwen3-0.6B
```

曾经打印过 `torch 2.11.0+cu128 / Triton 3.6.0`，那是安装过程中的中间状态，
不是后续正式实验环境，不能混入最终报告。

路径分属两个不同机器：

```text
Mac 本地仓库       /Users/yuki/Documents/nano-vllm/vllm
服务器仓库         /nano-vllm/learn-nano-vllm
服务器模型         /nano-vllm/models/Qwen3-8B
服务器实验         /nano-vllm/5090-runs/v3
```

Mac 路径不能在服务器容器中使用。对话末期发生的
`cd /Users/yuki/...: No such file or directory` 就属于主机路径混用，不是 Git
仓库损坏。

## 4. 环境、网络与服务器问题全记录

### 4.1 GitHub clone/fetch/pull 超时

**现象**：`git clone` 连接 GitHub 443 端口约 129 秒后超时；后续
`git fetch origin`、`git pull --ff-only` 也多次卡住。

**分类**：外部网络问题，不是项目代码问题。

**已使用的稳定做法**：

```bash
timeout 60s git \
  -c http.version=HTTP/1.1 \
  fetch -4 \
  --no-tags \
  --filter=blob:none \
  origin codex/slo-aware-scheduler

git merge --ff-only FETCH_HEAD
```

**注意**：只有 fetch 成功后才能 merge；建议用 `&&` 连接，避免合并旧的
`FETCH_HEAD`。

### 4.2 远端分支不存在

**现象**：

```text
fatal: cannot set up tracking information;
origin/codex/slo-aware-scheduler is not a branch
```

**原因**：当时本地分支尚未成功推送或服务器尚未 fetch 到该远端 ref。

**状态**：已解决。当前远端分支存在，并已包含 `2955070`。

分支切换时还曾出现 `D master_nano-vllm-v2.md`。这表示服务器工作树里该文件
当时处于删除状态，与“远端分支不存在”是两件事。当前仓库该文件存在；以后
切分支前应先 `git status --short`，不要把用户工作树改动误当成远端问题。

### 4.3 Hugging Face Xet/CAS 401

**现象**：下载 Qwen3-0.6B 到 70%--80% 时，访问
`cas-server.xethub.hf.co` 返回 `401 Unauthorized`。

**分类**：Hugging Face Xet/CAS 下载链路问题，不是模型权限或推理代码问题。

**处理事实**：对话中降低到 `--max-workers 2` 后继续下载，0.6B 和 8B 最终都
完成；用户明确要求以后 Hugging Face 资源优先使用镜像。

**固定偏好**：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

若镜像缺文件，再回官方源；不要同时让模型下载、PyTorch wheel 和 GitHub 大
wheel 抢同一条受限链路。

### 4.4 PyTorch/Triton wheel 下载极慢

**现象**：`torch` 约 502MiB、`triton` 约 188MiB 长时间停在 preparing；
阿里云 PyTorch wheel 索引也曾卡在 `torch==2.2.0`。

**结论**：索引可连接不代表目标 wheel 在该镜像存在或链路足够快。对
`download.pytorch.org` 执行根路径 HEAD 得到 403，只能说明根路径不允许目录
访问，不能据此判定具体 wheel URL 不可用。

**最终状态**：正式环境固定为 PyTorch 2.8.0+cu128、Triton 3.4.0。

### 4.5 PyTorch CUDA runtime 与 nvcc 的区别

**现象**：PyTorch 显示 CUDA 12.8 可用，但 `nvcc --version` 报
`command not found`。

**事实**：PyTorch wheel 自带运行所需 CUDA runtime，不包含完整 CUDA
Toolkit 编译器。运行 PyTorch、Triton 和预编译 FlashAttention wheel 不要求
系统 `nvcc`；以后编译自定义 `.cu` extension 才需要安装匹配 Toolkit。

这不是“PyTorch 没装好”。

### 4.6 FlashAttention wheel 下载/SCP 失败

**现象**：GitHub 244MiB wheel 下载只有约 20KiB/s；断点续传停在 12MiB，
随后连接超时；从 Mac `scp -P 32222` 又因密码/连接问题失败。

**处理**：服务器端使用 `curl -L -C - --retry ...` 继续下载，最终依赖安装完成。

`ps` 中 curl 的 `STAT=S` 只表示进程在等待网络/事件；真正判断是否推进要同时看
wheel 文件大小/mtime 和下载日志。文件固定在 12MiB 且日志连续 timeout 时才是
链路停滞。

**分类**：网络与服务器 SSH 凭据问题，不是 FlashAttention ABI 问题。最终 wheel
与 Python 3.12、torch2.8、cu12、CXX11 ABI TRUE 匹配。

### 4.7 GPU 存在但 PyTorch/NVML 不可用

**现象**：

```text
/dev/nvidia6 存在
/proc/driver/nvidia 显示 RTX 5090
nvidia-smi: Failed to initialize NVML: Unknown Error
torch.cuda.is_available(): False
torch.cuda.device_count(): 0
```

驱动信息为 610.43.02，设备 minor 为 6。说明容器能看到部分设备节点，但用户态
NVML/CUDA 初始化失败。

**处理**：重启租用实例后恢复；PagedAttention CUDA 测试随即可以运行。

**结论**：这是实例/容器 GPU 映射或驱动状态故障，不应通过重装 PyTorch反复
“修复”。今后遇到同样组合，优先保存日志并重启实例。

### 4.8 CUDA 测试被 skipped

第一次运行 `tests.test_paged_attention` 时全部显示 `CUDA is required`，根因就是
上一节的 `torch.cuda.is_available() == False`。实例恢复后：

```text
test_gqa_non_aligned_context_for_all_page_sizes ... ok
test_mixed_query_positions ... ok
extended matrix ... skipped（需要显式环境变量）
```

`OK (skipped=...)` 只代表测试框架成功跳过，不代表 GPU correctness 已执行。

### 4.9 本地 CPU 环境无法 import Triton kernel

Mac 本地可以完成 scheduler/metrics/spec/block-manager 等纯逻辑测试和 compile
检查，但完整 `unittest discover` 会因为本机缺少 CUDA/Triton/FlashAttention
运行环境而无法覆盖 kernel 模块。最新一轮明确执行并通过的是 51 项非 CUDA
回归；GPU 测试必须看服务器日志，不能用本地 skipped 代替。

### 4.10 环境变量为空导致输出路径和模型路径错误

**现象**：

```text
tee: /paged/triton-page16.log: No such file or directory
assert os.path.isdir(self.model)
```

第一条说明 `$RUN_ROOT` 为空，拼出的路径退化为 `/paged/...`；第二条说明
`$MODEL_8B` 为空或指向不存在目录。这不是 attention kernel 错误。

每次新 shell/实例都要重新执行并检查：

```bash
export RUN_ROOT=/nano-vllm/5090-runs/v3
export MODEL_8B=/nano-vllm/models/Qwen3-8B
mkdir -p "$RUN_ROOT/paged"
test -d "$MODEL_8B"
```

### 4.11 当前服务器问题：尚未提供错误日志

用户目前只说明“服务器那边出现问题”，尚未贴出 graph smoke 的 traceback、
OOM、illegal memory access、卡住位置或 NVML 状态。因此本文不会伪造根因。

与最新提交相关的已知高风险点是 Triton PagedAttention CUDA Graph 尚未经过
服务器成功验收，见第 10 节。它可能与问题相关，但在拿到日志前不能下结论。

### 4.12 原始实验产物仍主要留在服务器

仓库 `.gitignore` 排除了 `runs/`，当前 Git 中没有 benchmark JSON/log。提交到
仓库的是结果表和解释，而不是全部原始 trace。若租用实例磁盘随故障或到期被
回收，held-out、paged 和 split-K 的原始证据会丢失，只剩文档中的汇总数字。

因此服务器恢复后的第一优先级之一是把 `/nano-vllm/5090-runs/` 整体备份到
持久存储；随后可以选择一小组不含模型权重的代表性 JSON 纳入版本化 artifact，
或至少保存校验和和文件清单。

## 5. 从 baseline 到当前分支：实际实现清单

### 5.1 上游已经有，不能算作本项目新增

`origin/main` 已提供 Qwen3、prefill/decode、continuous batching、paged KV、
prefix cache、chunked prefill、FlashAttention、decode CUDA Graph、Tensor
Parallel 和 temperature sampling。

本项目不能在简历中写“从零实现 vLLM/Paged KV/CUDA Graph”。准确说法是：
在 nano-vLLM baseline 上重构调度、观测和 attention backend，并实现实验性
细 page、mixed batch 与 speculative 路径。

### 5.2 请求级观测与 benchmark（`CODE + UNIT + E2E`）

主要入口：

- [`nanovllm/engine/metrics.py`](../nanovllm/engine/metrics.py)
- [`benchmarks/benchmark_slo.py`](../benchmarks/benchmark_slo.py)
- [`benchmarks/compare_results.py`](../benchmarks/compare_results.py)

实现内容：TTFT、TPOT、E2E、queue、prefill/decode time、ITL、Max ITL、
preemption、prefix hit、KV 分配/有效/尾块浪费、step phase、GPU 显存、
proposed/accepted token、arrival pattern、固定 seed 和环境 provenance。

benchmark JSON 当前为 schema v3，记录 Git commit、Torch/CUDA/Triton/
FlashAttention、模型摘要和逐请求 token IDs。

### 5.3 `slo_aware` v1（`REJECTED`）

v1 引入 prefill/decode 交错、固定较小 chunk、decode rotation 和连续 decode
上限。0.6B bulk 实验中：

| 指标 | prefill-first | v1 |
|---|---:|---:|
| TTFT P95 | 602.08ms | 1101.29ms |
| E2E P95 | 2061.78ms | 2351.17ms |
| Output tok/s | 2064.04 | 1765.36 |
| Prefill chunks P95 | 1 | 4 |

固定小 chunk 增加启动开销并延迟后续请求准入。该失败直接推动 v2，不应删除。

### 5.4 `slo_aware_v2`（`ACCEPTED`，当前最强成果）

主要入口：[`nanovllm/engine/scheduler.py`](../nanovllm/engine/scheduler.py)。

实现内容：

- waiting/request 使用 TTFT deadline；running/request 使用 TPOT deadline；
- 用 normalized slack 比较两个不同 SLO 尺度；
- waiting 内部 least-laxity-first；
- EWMA 估计 prefill token cost 和 decode step cost；
- 根据 decode slack 动态决定 prefill chunk；
- round-robin decode 和可解释抢占。

0.6B Poisson 4/8/12/16 req/s 扫描中，v2 的 Max ITL P95 保持在
24--33ms，违反率约 1%，而 prefill-first 在高负载升至 905.71ms/84.9%。

8B 灵敏度实验修正了一个重要的报告问题：最初比较表随每行配置使用不同 TPOT
阈值，导致 violation rate 不可横向比较。随后 `compare_results.py` 增加固定的
`--eval-ttft-slo-ms` 和 `--eval-itl-slo-ms`，最终 held-out 统一使用
500ms/75ms。

### 5.5 Prefix cache 证据（`E2E`）

8B、Poisson 2 req/s、shared prefix 512：

| 配置 | TTFT P95 | Output tok/s | Cache hit |
|---|---:|---:|---:|
| v2 无共享前缀 | 211.59ms | 250.53 | 0.00 |
| v2 prefix512 | 95.14ms | 251.16 | 0.50 |

这是 prefix reuse 改善 TTFT 的正实验，但不是本项目从零新增 prefix cache。

### 5.6 Triton KV-store（`GPU-CORRECT + MICRO + E2E`）

[`store_kvcache`](../nanovllm/layers/attention.py) 支持非 2 的幂宽度，D=768/1024、
token=1/8/64/512/4096 的微基准相对 PyTorch 为 1.73--2.90 倍。

按 Qwen3-8B 36 层预测每 decode step 节省 0.2524ms；E2E 实测 TPOT
`17.71 -> 17.44ms`，节省 0.27ms，方向和数量级吻合。Output tok/s 只提高
0.07%，因为 Poisson 2 req/s 下它主要受 offered load 限制。

### 5.7 Triton RMSNorm（`GPU-CORRECT + MICRO + REJECTED`）

Triton 相对 eager 快 3.2--8.7 倍，但 baseline 已使用 `torch.compile`。
在真实 decode 形状上胜负混合，E2E compiled/triton 的 output tok/s
`244.40/244.41`，TPOT `17.38/17.44ms`。因此默认保持 compiled，Triton 仅作
学习和大 prefill 研究后端。

### 5.8 课程、学习和面试材料（工程表达产出）

提交 `3d94882`、`6dd43de` 增加项目定位、中文学习导读、面试讲稿和分模块课程。
这是交付的一部分，但应与核心推理性能实现分开陈述。

## 6. v3 统一调度与 mixed batch

### 6.1 接口重构（`CODE + UNIT`）

[`ScheduledRequest`](../nanovllm/engine/outputs.py) 为每个请求保存：

```text
num_scheduled_tokens
is_prefill
needs_sampling
speculative_token_ids
speculative_method
accepted_tokens
```

[`SchedulerOutput`](../nanovllm/engine/outputs.py) 汇总 prefill/decode token，不再
依赖全局 phase；`from_phase()` 和 `__iter__()` 为旧策略保留兼容入口。

需要诚实说明：旧策略有接口层和调度单测，但仓库没有保存完整 old-policy trace
或所有旧 CLI 的回归快照，因此“兼容”是代码/单测级，不是完整行为证明。

### 6.2 `slo_aware_v3`（`CODE + UNIT`，未 `ACCEPTED`）

v3 实现：

- decode 候选优先占用 token/sequence budget；
- waiting 请求按 normalized TTFT slack 排序；
- 同一步允许多个 partial prefill；
- mixed 成本使用 prefill-token/decode-request 的 power-of-two 二维 bucket；
- 未见 bucket 回退到 pure prefill + pure decode EMA；
- KV 不足时选择 normalized decode slack 最大的 victim；
- prefill chunk granularity 与 KV page size 解耦。

这些机制分别有 scheduler 单测。但现有 v3 mixed-length 结果没有满足原路线的
最终门槛，不能写“v3 已优于 v2”。

### 6.3 BatchMetadata 与 mixed varlen（`CODE + UNIT/GPU-CORRECT`）

[`BatchMetadata`](../nanovllm/utils/context.py) 保存 `cu_seqlens_q/k`、
`context_lens`、`slot_mapping`、`block_tables`、`logits_indices`、
`query_to_request` 和 `query_positions`。

ModelRunner 将 decode qlen=1、chunked prefill 和 speculative verify 打包进同一
varlen forward；partial prefill 不采样，LM Head 只处理 `logits_indices` 指定
位置。Flash mixed attention 已与 PyTorch reference 做容差对比。

## 7. BlockPool、prefix 与 speculative rollback

[`BlockManager`](../nanovllm/engine/block_manager.py) 已实现：

- intrusive `FreeBlockQueue`，O(1) touch/free/LRU；
- refcount；
- 同 hash 多物理 block；
- token 内容二次校验，防 hash collision；
- committed full-block prefix hash；
- `reserve()` 和 `truncate()`；
- allocated/reserved/computed/uncomputed/tail-waste 分离统计。

单测覆盖 LRU、refcount、collision、fail-closed reserve、truncate、共享前缀 240
在 block16 命中而 block256 不命中，以及细 page 尾块浪费至少降低 8 倍。

这里的“8 倍”是合成状态单测，不等于已经在 Qwen3-8B 在线 workload 上完成
内存收益验收。

## 8. Triton PagedAttention

### 8.1 General kernel（`GPU-CORRECT + E2E REJECTED`）

[`AttentionBackend`](../nanovllm/layers/attention.py) 抽象保留 Flash reference，
`triton_paged` 支持 page16/32/64、Qwen3 GQA、logical-to-physical block mapping、
严格 causal mask 和 FP32 online softmax。

对话中贴出的 general kernel matrix 覆盖 page16/32/64、batch1/8/32/128、
context128/512/2048/4096，最大误差小于 0.002。实例恢复后 smoke GPU 单测也通过
GQA 非对齐 context 和 mixed query position；扩展 unittest 需要显式环境变量，
不能把 skipped 当作已执行。

Eager mixed-length E2E：

| 指标 | Flash block256 | Triton page16 | Triton page32 |
|---|---:|---:|---:|
| Output tok/s | 267.24 | 222.37 | 224.66 |
| TTFT P95 | 5428.97ms | 12481.60ms | 11998.74ms |
| TPOT P95 | 27.92ms | 38.19ms | 37.89ms |
| E2E P95 | 8935.55ms | 16226.45ms | 15699.97ms |

page32 只有 Flash eager 吞吐的 84.1%，未达到 95% 目标。两者 offered output
load 约 256 tok/s；Triton 低于输入负载，TTFT/queue 的巨大值主要反映队列不
稳定，不能直接当作单次 attention latency。

### 8.2 split-K（`GPU-CORRECT + MICRO + E2E REJECTED`）

split-K 把 context 按 512 token 分区：第一阶段写 FP32 partial max/sum/acc，
第二阶段归并全局 softmax。workspace 每层 grow-only 缓存。

batch=8、page32：

| Context | 热缓存 speedup | 256MiB flush 后 speedup |
|---:|---:|---:|
| 512 | 0.499x（强制 split） | 1.091x |
| 2048 | 1.525x | 1.904x |
| 4096 | 2.133x | 1.993x |

冷缓存结果证明设备端并行化收益真实存在。但 eager E2E：

| 指标 | page32 general | page32 auto split-K |
|---|---:|---:|
| Output tok/s | 221.81 | 203.20 |
| TPOT P95 | 38.06ms | 41.38ms |
| TTFT P95 | 12477.93ms | 16814.19ms |
| E2E P95 | 16205.63ms | 21050.43ms |
| Peak GiB | 27.54 | 27.66 |

吞吐下降 8.4%，所以 `general` 已恢复为安全默认值。split-K 路径保留为 opt-in
负实验。阶段数据中 batch8 占 pure-decode step 的 89%，auto pure-decode
平均约 28.053ms。这里的 `model_ms` 基于 host `perf_counter` 且没有独立 GPU
event，同步边界会受 sampling `.tolist()` 影响，因此只能用于相同 harness 的
归因，不能称为纯 GPU latency。

## 9. Greedy 结果与可复现性问题

在线 greedy 对比最初出现 v2/v3 多请求 token 分叉；随后 baseline v2 与自身
重跑也有 9/100 请求分叉。固定隔离条件下 v2 与 v3 token IDs 全部一致。

因此项目得出的正确结论不是“greedy 天生不可复现”，而是：

- 固定 seed 不能保证不同动态 batch shape 的 BF16/GEMM/attention reduction
  位级一致；
- top-2 logit margin 很小时，微小数值差异可以翻转 argmax；
- baseline 自身不稳定时，不能把所有跨策略 token diff 都判成 mixed batch bug；
- isolated committed-KV/token 不变量、reference 容差、状态单测和首分叉 logit
  diagnostics 要一起使用。

`compare_greedy_outputs.py` 和 `--record-token-diagnostics` 已加入，但 diagnostics
会增加 top-k 和 CPU 同步，不能用这类运行报告性能。

## 10. CUDA Graph：最新实现与当前高风险问题

提交 `2955070` 将 `use_cudagraph` 从“仅 Flash”放宽为所有非
`--enforce-eager` pure decode，并为 Triton graph capture/replay 增加静态
`query_to_request` 和动态更新的 `query_positions`。

状态必须写成：

```text
CODE:        是
UNIT:        无专门 graph 集成测试
GPU smoke:  尚未看到成功输出
E2E:         未运行/未确认
ACCEPTED:    否
```

### 10.1 Code review 发现的 auto 路由偏差

capture 时 `block_tables` 宽度固定为 `max_model_len / page_size`。split-K 的
partition 数由 block table 宽度计算，而不是由 replay 时真实
`query_positions/context_lens` 计算。以 max_model_len=4096、page32 为例，
capture 会得到 8 个 512-token partition。

因此 graph bucket 的 batch 小于 32 时，`auto` 会在 capture 阶段固定选择
split-K；即使 replay 的真实 context 只有 128/512，也不会像 eager auto 那样
回退 general。这与旧文档“context > 512 才启用 split-K”的描述不一致。

它不必然导致崩溃，但可能造成性能偏差、额外 workspace 和无效 partition。
当前服务器问题尚无日志，不能直接断言它就是根因。

### 10.2 恢复服务器后的安全顺序

1. 先确认 `nvidia-smi`、`torch.cuda.is_available()` 和提交号；
2. 保存 graph smoke 完整 log，而不是只报告“卡住”；
3. 先用 `--paged-attention-decode-kernel general` 验证 Triton graph 基础路径；
4. 再显式运行 `auto`；
5. graph 失败时用 `--enforce-eager --paged-attention-decode-kernel general`
   回到已经验证的安全路径；
6. 在 graph smoke 和 token correctness 通过前，不继续做完整 E2E 或 GQA 融合。

## 11. Speculative decoding：实现了什么，缺什么

### 11.1 已实现（`CODE + UNIT`）

- greedy-only 配置校验，非零 temperature 拒绝 speculative；
- n-gram 最长后缀匹配，无匹配回退普通 decode；
- target 一次验证 proposals 并产生 bonus；
- 全接受、首 token 拒绝、部分接受；
- EOS/max_tokens 截断；
- KV `reserve/truncate` 与 committed length 回滚；
- Qwen3-0.6B draft 模型加载；
- target + draft KV block bytes 联合规划；
- draft 仅支持 TP=1。

### 11.2 尚未验收（`UNVERIFIED`）

- 固定 100 条请求与非 speculative target token IDs 完全一致；
- n-gram 普通 workload 的 acceptance/TPOT；
- Qwen3-8B + 0.6B 在 32GB 内完整运行证据；
- draft 高接受率 workload TPOT 改善至少 10%；
- draft 普通 workload 回退幅度；
- speculative 与 fine-page PagedAttention/CUDA Graph 的组合。

所以面试中只能说“实现了 greedy speculative MVP 和回滚状态机”，不能说
“推测解码已加速 10%”。

## 12. 测试审计

仓库定义了 64 个 unittest 方法：

| 领域 | 测试数 | 备注 |
|---|---:|---|
| Scheduler | 22 | v1/v2/v3/spec scheduling |
| Benchmark | 8 | arrivals、阈值、default backend |
| BlockManager | 8 | LRU/refcount/hash/truncate/fragmentation |
| Spec decode | 6 | proposer 与 greedy verify |
| PagedAttention | 5 | CUDA gated，extended matrix opt-in |
| Metrics | 4 | lifecycle/ITL/preemption/diagnostics |
| Greedy compare | 4 | token diff 与 margin diagnostics |
| RMSNorm | 3 | 2 项 CUDA gated |
| Outputs | 2 | mixed aggregate/worker state |
| KV-store | 1 | CUDA gated |
| Compare results | 1 | fixed evaluation threshold |

最新本地明确执行的是 51 项非 CUDA 回归，全部通过。服务器曾明确执行并通过
PagedAttention GQA/mixed smoke；split-K micro 每个 case 内置 reference assert。
但没有对话证据证明当前 `2955070` 的 CUDA Graph 路径已通过任何专门测试。

## 13. 双轴代码审查

### 13.1 Standards

仓库没有独立 `CODING_STANDARDS.md`、`CONTRIBUTING.md` 或 AGENTS 规则；
`pyproject.toml` 也没有额外 lint/style 约束。本次没有发现硬性仓库标准违规。

判断项：

1. **文档状态过期（高）**：旧复习手册仍把 Triton CUDA Graph 写成未来工作，
   但 `2955070` 已有实现；diff 统计也已过期。
2. **Repeated Switch / Primitive Obsession（低）**：`general/split_k/auto` 字符串
   在 Config、Attention 和两个 benchmark 重复声明，未来容易漂移。
3. **Divergent Change（低）**：`model_runner.py` 同时承担分布式生命周期、batch
   preparation、draft、metrics 和 CUDA Graph；下一次 graph 扩展前应考虑抽出
   graph state 模块。

### 13.2 Spec

1. **阶段 0/5090 验收未闭环**：v2 数据较完整，但 v3 的 5% 改进、backend 95%
   和 speculative 验收没有完成。
2. **CUDA Graph auto 语义偏差**：按静态最大 block table 而非真实 context 选择
   partition，短 context 也可能固定走 split-K；无 graph 集成测试。
3. **阶段 3 仅部分完成**：代码/单测存在，100 条 token equality、acceptance、
   TPOT、32GB 和 10% 目标缺失。
4. **旧策略兼容证明不完整**：有适配接口和单测，没有完整 trace/CLI snapshot。
5. **额外交付**：RMSNorm、KV-store、课程和面试文档超出 v3 核心路线，但没有
   破坏核心实现；报告中应作为实验/工程表达分开列出。

审查汇总：Standards 3 个判断项、0 个硬违规，最严重是文档状态过期；Spec
5 个发现，最严重是未验证的 CUDA Graph auto 路由与验收缺口。

## 14. 现有文档需要纠正的口径

| 旧说法 | 审计后的说法 |
|---|---|
| 当前改动 60 files/7751 insertions | 截至 2955070 为 61/9521/351 |
| CUDA Graph 只支持 Flash，Triton 是未来工作 | Triton graph 已实现，但未通过服务器验收 |
| split-K 下一步待实现/待跑 E2E | 已实现；micro 快、eager E2E 吞吐下降 8.4% |
| PagedAttention 已完成 | general correctness 完成，性能未达 95%；graph 未验 |
| speculative decoding 已完成 | MVP/单测完成，GPU acceptance/E2E 未完成 |
| greedy 金标应始终 bit-exact | 动态 batch 下 baseline 自身也可能分叉；需要 margin 诊断 |
| model_ms 就是 GPU kernel latency | 当前主要是 host timing，不能当独立 GPU event |

## 15. 面试时可以说与不能说

### 可以说

- 在 nano-vLLM baseline 上实现了请求级 SLO 指标和 v1/v2/v3 策略；
- v2 held-out 将 Max ITL P95 降低 73.7%，吞吐下降 0.16%，并诚实报告 TTFT/E2E
  代价；
- 重构为 per-request token scheduler，支持 mixed decode + multi-partial-prefill；
- 实现 BlockManager 的 O(1) free/LRU、collision check 和 speculative truncate；
- 自研 Triton fine-page PagedAttention 和 split-K，并用 micro/E2E 发现局部加速
  不等于系统加速；
- 实现 greedy n-gram/draft MVP，但 GPU 性能验收尚未完成。

### 不能说

- “从零实现了完整 vLLM”；
- “v3 已经全面优于 v2/FlashAttention”；
- “PagedAttention 达到了 Flash 的 95%”；
- “split-K 已经端到端加速”；
- “CUDA Graph Triton 路径已经稳定”；
- “draft speculative 已经快 10%”；
- “多卡、HTTP 服务、FP8 KV、量化已经完成”。

## 16. 当前恢复优先级

```text
P0  获取当前服务器问题的完整日志并恢复 GPU 可用性
P0  备份 /nano-vllm/5090-runs，防止租用实例回收原始证据
P0  对 2955070 先跑 graph-general smoke，再跑 graph-auto
P1  修复/明确 graph auto 的运行时 context 路由
P1  增加 CUDA Graph 集成正确性测试和 eager/graph token 对比
P1  只有 graph E2E 通过后再决定 split-K 是否重新启用
P2  完成 n-gram/draft 的固定 100 请求、acceptance、TPOT、显存验收
P2  统一 decode-kernel 枚举，降低配置漂移
P3  抽离 ModelRunnerGraphState，再考虑 GQA KV group 复用
```

项目目前最有价值的并不是“所有优化都成功”，而是已经形成了一条可以解释失败
的证据链：

```text
定义用户指标
  -> 建立可复现 workload
  -> 单元/数值正确性
  -> microbenchmark
  -> E2E A/B
  -> 达标才默认启用，否则保留负实验与安全回退
```
