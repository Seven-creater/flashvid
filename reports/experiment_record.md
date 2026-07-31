# 实验一：FlashVID Vision Encoder 压缩复现与 vLLM 部署

## 1. 任务要求

### 1.1 总目标

- 联网调研论文 [FlashVID](https://arxiv.org/abs/2602.08024) 及其[公开代码](https://github.com/Fanziyang-v/FlashVID)。
- 在 `Qwen/Qwen3.5-4B` 上复现 FlashVID 的 Vision Encoder/Before-LLM Compression。
- 只实现视觉侧压缩，不实现 LLM 内部 token pruning。
- 将实现改造成 vLLM 0.25.1 out-of-tree 插件，保留标准 OpenAI API。
- 增加服务级参数 `--vision-retention-ratio`，控制视觉 token 保留比例。
- 使用 8 张 A6000 做数据并行推理，比较 DP=1/2/4/8 并选择实测最快配置。
- 本地开发后上传 [Seven-creater/flashvid](https://github.com/Seven-creater/flashvid)，再在服务器工作区拉取、安装和测试。
- 不修改已有数据集和模型目录；新增环境、模型、缓存、日志和结果全部放到项目目录。

### 1.2 压缩任务

本实验只包含一种压缩方式：Vision Encoder 输出后、LLM prefill 前的视觉 token 压缩。
该方式内部包含：

```text
DySeg
  -> ADTS 选择重要视觉 token
  -> TSTM 合并时空冗余 token
  -> DPC-kNN 空间上下文聚合
  -> 更新 visual embeddings / placeholders / M-RoPE
  -> LLM prefill
```

固定参数：

| 参数 | 值 |
|---|---:|
| ADTS `alpha` | 0.7 |
| temporal threshold | 0.8 |
| segment threshold | 0.9 |
| minimum segments | 4 |
| token selection | `attn_div` |
| Inner-LLM expansion | 不使用 |

明确不实现：`fastv_prune`、`pruning_layer`、`llm_retention_ratio` 和任何 LLM 层内部
token pruning。

### 1.3 部署与验收任务

- 自定义架构：`FlashVIDQwen3_5ForConditionalGeneration`。
- 参数范围：`0 < vision_retention_ratio <= 1`。
- `ratio=1.0` 必须精确绕过压缩，作为原始模型基线。
- 支持纯文本、图片、单视频和多视频请求。
- 测试比例：`1.0 / 0.5 / 0.25 / 0.1`。
- 测试本地算法、服务器插件、真实权重 API、原生基线和并行吞吐。
- 最终推荐部署为 `DP=8、TP=1`，前提是服务器实测无 OOM、无失败请求。

## 2. 实际做法与进展顺序

### 第一步：调研论文、上游实现和 vLLM

确认 FlashVID 论文包含 Before-LLM 和 Inner-LLM 两类压缩。本实验按要求只复现
Before-LLM 路径，但保留构成该路径的 ADTS、TSTM、DySeg 和 DPC-kNN。

随后检查 vLLM 0.25.1 的 Qwen3.5/Qwen3-VL 多模态实现，确认其已经提供动态视频占位
token、M-RoPE 重算和多模态插件注册接口，因此采用 out-of-tree 模型插件，而不是
复制或修改整个 vLLM。

### 第二步：实现独立视觉压缩模块

在 `src/flashvid/compression.py` 中实现：

- 动态时间分段 DySeg。
- attention-and-diversity token selection。
- TSTM 树式时空合并。
- DPC-kNN 空间上下文聚合。
- 精确目标预算对齐和 anchor 索引输出。
- `ratio=1.0` 原张量绕过路径。

压缩模块只依赖 PyTorch，可脱离 vLLM 单独做固定张量测试。

### 第三步：接入 Qwen3.5 Vision Encoder

在 Qwen3.5 Vision Encoder 最后一层 QKV 投影上注册临时 hook，提取逐帧 query/key，
计算每个视觉 token 接收到的注意力。注意力矩阵按帧批量计算，并用 query chunk 控制
峰值显存，避免逐帧 Python 串行成为主要瓶颈。

压缩发生在 vision merger 输出后、LLM prefill 前。每个合并 token 继续使用 anchor
索引更新：

- 视频占位 token 数量和顺序。
- M-RoPE 时间与空间位置。
- 所有拼接的视觉特征通道。
- DeepStack 特征通道兼容路径。

Qwen3.5-4B 当前官方配置没有启用 DeepStack 层，但实现没有将该情况写死。

### 第四步：实现 vLLM 插件和服务参数

注册自定义架构 `FlashVIDQwen3_5ForConditionalGeneration`，并实现
`flashvid-serve` 启动器。启动器将：

```text
--vision-retention-ratio R
  -> FLASHVID_VISION_RETENTION_RATIO=R
  -> vLLM video_pruning_rate=1-R
  -> placeholders 与压缩后 embeddings 保持同步
```

服务仍使用标准 `/v1/chat/completions`，不增加请求级私有字段。图片和纯文本沿用原始
路径；视频进入 FlashVID 压缩路径。为保持视频时间轴有效，每个采样时间位置至少保留
一组空间 token。

### 第五步：本地测试与范围检查

建立 ADTS、TSTM、DPC-kNN、预算、ratio-1.0 bypass 和输入合法性测试，并增加源码范围
检查，防止 Inner-LLM pruning 逻辑进入实现。

本地结果：

```text
16 passed
```

### 第六步：上传 GitHub并准备服务器工作区

代码在本地目录完成：

```text
C:\Users\29785\Desktop\flashvid
```

上传仓库：

```text
https://github.com/Seven-creater/flashvid
```

服务器项目目录：

```text
/data02/usr/wangqihao/Demo/test/flashvid
```

虚拟环境、模型、缓存、日志和结果分别放在 `.venv/`、`models/`、`.cache/`、`logs/`
和 `results/`。没有修改 `/data02/pretrained_model/cvr_learn`。

### 第七步：安装 vLLM、CUDA 兼容环境和真实模型

服务器环境为 8 张 NVIDIA A6000，主机驱动报告 CUDA 12.2；项目安装
`vllm==0.25.1` 和对应的 PyTorch CUDA 13 wheel。Qwen3.5-4B 从 ModelScope 下载到：

```text
/data02/usr/wangqihao/Demo/test/flashvid/models/Qwen3.5-4B
```

模型约 8.8 GB，两个 safetensors 分片均完整。CUDA forward-compat runtime、CUDA
toolkit 和全部 JIT 缓存均限定在项目目录。

### 第八步：先做 dummy 权重端到端验证

先使用 `--load-format dummy` 启动单卡服务，验证插件注册、模型构造、Vision Encoder、
压缩、占位 token、M-RoPE 和生成链路。dummy 服务成功返回视频请求 HTTP 200 后，再
切换真实权重，避免每次错误都重复加载模型。

### 第九步：真实权重功能与比例测试

真实 Qwen3.5-4B 依次验证纯文本、图片、单视频和双视频请求。随后分别启动
`ratio=1.0 / 0.5 / 0.25 / 0.1` 服务，记录相同视频的 prompt token 数。

最后单独启动原生 vLLM Qwen3.5 服务，与 ratio-1.0 插件做 greedy 输出对照。两者
prompt token 数和生成内容一致。

### 第十步：DP=1/2/4/8 吞吐测试与后台部署

使用相同视频、`temperature=0` 和固定生成长度比较 DP=1/2/4/8。所有服务通过
`nohup + setsid` 后台启动，等待 `/health` 返回 200 后再运行 benchmark；切换配置时
按进程组停止旧服务，避免残留 EngineCore 占用端口或显存。

DP8 吞吐最高，因此最终后台服务使用：

```text
DP=8, TP=1, max_num_seqs=64, max_num_batched_tokens=32768
```

## 3. 出现的问题与解决过程

### 3.1 SSH 别名不可用

**现象：** 原计划中的 SSH 别名无法解析，不能进入服务器工作区。

**解决：** 通过现有连接信息确认服务器地址为 `10.1.4.86`，后续统一使用该地址。

**验证：** 可以在 `/data02/usr/wangqihao/Demo/test/flashvid` 完成 clone、pull、安装和测试。

### 3.2 主机驱动与 PyTorch CUDA 13 runtime 不匹配

**现象：** PyTorch 初始化 GPU 时报 `NVIDIA driver is too old`。主机驱动只直接支持
CUDA 12.2，而 vLLM wheel 使用 CUDA 13 runtime。

**根因：** wheel 所需用户态 CUDA 版本高于主机驱动直接暴露的版本。

**解决：** 在项目 `.venv` 内安装 `cuda-compat=13.0.2`，启动器设置项目级
`LD_LIBRARY_PATH`；不升级系统驱动，不需要 root。

**验证：** PyTorch 可以识别 A6000，并成功执行 GPU 矩阵运算。

### 3.3 FlashInfer 错用系统 CUDA 11.8 nvcc

**现象：** GPU 初始化成功后，FlashInfer JIT 仍报告 CUDA 编译版本错误。

**根因：** JIT 从系统 PATH 找到了 CUDA 11.8 的 nvcc，而不是 wheel 附带的 CUDA 13
toolkit。

**解决：** `flashvid-serve` 显式设置 `CUDA_HOME`、`FLASHINFER_NVCC` 和 PATH，使
FlashInfer/Torch JIT 使用项目 `.venv` 内的 CUDA 13 toolkit。

**验证：** FlashInfer sampler 编译越过 nvcc 11.8 错误。

### 3.4 CUDA 13.0 headers 与 CUDA 13.2 nvcc 不一致

**现象：** 切换到 wheel nvcc 后仍出现 `CUDA_VERSION` mismatch。

**根因：** 已安装 runtime headers 为 13.0，nvcc 为 13.2。

**解决：** 在 `setup_server.sh` 固定 `nvidia-cuda-runtime==13.2.86`，使编译器和
headers 对齐。

**验证：** CUDA 版本检查通过，JIT 进入链接阶段。

### 3.5 FlashInfer 找不到 `libcudart.so`

**现象：** JIT 链接时报找不到 `-lcudart`。

**根因：** wheel 提供 `libcudart.so.13` 和 `lib/`，部分构建逻辑查找
`lib64/libcudart.so`。

**解决：** 仅在项目虚拟环境内创建：

```text
nvidia/cu13/lib64 -> lib
lib/libcudart.so -> libcudart.so.13
lib/libcuda.so -> .venv/cuda-compat/libcuda.so
```

**验证：** dummy 和真实权重 vLLM 服务均完成 FlashInfer warmup 并启动 API server。

### 3.6 停止父进程后端口仍被占用

**现象：** 结束 `flashvid-serve` 父进程后，新服务报 `Address already in use`；GPU 上仍有
EngineCore 进程。

**根因：** vLLM API server 和 EngineCore 是子进程，只结束父进程不能完整清理。

**解决：** 使用 `setsid` 创建独立进程组，停止时按 PGID 发送 TERM，再用
`ss -ltnp` 和 `nvidia-smi` 确认端口、显存已经释放。

**验证：** 后续 ratio 和 DP 配置均可顺序切换，没有端口冲突。

### 3.7 原生 vLLM 基线没有继承项目 CUDA 环境

**现象：** 直接运行 `.venv/bin/vllm` 做原生 ratio-1.0 对照时，再次回退系统 CUDA
11.8，服务启动失败。

**解决：** 原生基线同样显式传递 CUDA toolkit、`FLASHINFER_NVCC` 和
`LD_LIBRARY_PATH`，只是不注册 FlashVID 自定义架构。

**验证：** 原生 Qwen3.5 服务启动成功；与 ratio-1.0 插件 prompt token 数和 greedy
生成内容一致。

### 3.8 DP 服务首次启动耗时较长

**现象：** DP=2/4/8 首次启动并行加载多份模型、编译 Torch/FlashInfer kernel；日志中
出现 shared-memory broadcast 等待提示，服务暂时没有监听端口。

**根因：** 其他 DP worker 仍在编译或进行 CUDA Graph capture，不是 OOM。

**解决：** 使用后台进程和独立日志，不做高频轮询；等待 `/health` 返回 200 后再运行
benchmark。所有编译缓存写入项目 `.cache` 供后续复用。

**验证：** DP2、DP4、DP8 最终全部启动，8 张 GPU 均加载成功，无 OOM。

### 3.9 自定义压缩统计日志没有显示

**现象：** vLLM 多进程日志能显示模型架构，但普通 Python logger 的 INFO 压缩统计没有
出现在服务日志。

**解决：** 改用 `vllm.logger.init_logger`，并把每个视频的压缩前后 token 数和耗时设为
可见的 warning 日志。

**验证：** 自定义模型架构和压缩统计均可从 vLLM 服务日志定位。

### 3.10 GitHub HTTPS 节点间歇不可达

**现象：** 多次 push/pull 遇到 connection reset、TLS terminated 或 443 timeout；默认
DNS 解析到的节点不可达。

**解决：** 本地先保留完整 commit；服务器同步使用 Git bundle 作为临时兜底；GitHub
恢复可达后再完成主分支 push，并让服务器仓库保持同一 commit。

**验证：** GitHub `main`、本地仓库和服务器工作区均包含最终实验文档与实现代码。

## 4. 简洁实验报告

### 4.1 主要结果

测试环境：Qwen3.5-4B、vLLM 0.25.1、8×NVIDIA A6000。

功能测试：

| 项目 | 结果 |
|---|---|
| 本地单元测试 | 16/16 通过 |
| 服务器单元测试 | 16/16 通过 |
| dummy 权重插件启动 | 通过 |
| 真实权重文本请求 | 通过 |
| 真实权重图片请求 | 通过 |
| 真实权重单视频请求 | 通过 |
| 同一请求双视频 | 通过 |
| ratio-1.0 原生基线对照 | 通过 |

相同视频的 prompt token 数：

| Vision retention ratio | Prompt tokens | 相对 ratio-1.0 |
|---:|---:|---:|
| 1.00 | 11,671 | 100.0% |
| 0.50 | 6,163 | 52.8% |
| 0.25 | 3,409 | 29.2% |
| 0.10 | 1,756 | 15.0% |

该视频只产生较少的采样时间位置，且实现要求每个时间位置至少保留一组空间 token，
所以低比例存在下限；更长视频会更接近请求比例。

并行吞吐：

| DP | 请求数 | 并发数 | 成功数 | Requests/s | Output tok/s | TTFT p50 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 8 | 16 | 1.81 | 57.95 | 2.59 s |
| 2 | 16 | 8 | 16 | 1.65 | 52.74 | 3.83 s |
| 4 | 32 | 16 | 32 | 2.63 | 84.24 | 4.86 s |
| **8** | 32 | 16 | 32 | **3.29** | **105.15** | 2.95 s |

DP8 为实测最快配置。启动后每张卡约使用 40.8 GiB（模型、运行时和缓存合计），所有
请求无 OOM、无失败。

### 4.2 结论

- 已完成 FlashVID Vision Encoder 压缩在 Qwen3.5-4B 上的工程复现。
- 实现包含 ADTS、TSTM、DySeg 和 DPC-kNN，不包含任何 Inner-LLM pruning。
- `--vision-retention-ratio` 可以控制服务级视觉压缩，`1.0` 精确绕过压缩。
- ratio-1.0 插件与原生 Qwen3.5 的测试 prompt token 数和 greedy 输出一致。
- 文本、图片、单视频和多视频均可通过标准 OpenAI API 调用。
- 8×A6000 上推荐 `DP=8、TP=1`，实测吞吐为 105.15 output tokens/s。
- 当前项目和模型目录：

```text
/data02/usr/wangqihao/Demo/test/flashvid
/data02/usr/wangqihao/Demo/test/flashvid/models/Qwen3.5-4B
```

推荐后台启动命令：

```bash
cd /data02/usr/wangqihao/Demo/test/flashvid
nohup setsid bash scripts/serve_dp8.sh 0.10 \
  > logs/serve_dp8-background.log 2>&1 < /dev/null &
```

### 4.3 尚未完成

1. 没有运行论文全部五个数据集的准确率评测；当前结论是工程复现和部署验证，不是完整论文指标复现。
2. 原生模型与 ratio-1.0 已做 greedy 内容对照，但没有导出逐 token 完整 logits 做数值级比较。
3. 当前吞吐使用重复视频，可能命中 vLLM 多模态缓存；尚未做大规模不同视频的 cold-cache 压测。
4. 没有系统扫描视频采样帧数与 retention ratio 对准确率的联合影响。
5. 当前压缩比例是服务启动参数，不支持同一服务内按请求动态改变比例。

因此，当前结果可以证明视觉压缩插件能够在真实 Qwen3.5-4B 和 vLLM 0.25.1 上稳定
部署，并验证视觉 token 减少和 DP8 服务吞吐；不能将其表述为论文全部数据集上的最终
精度复现结果。
