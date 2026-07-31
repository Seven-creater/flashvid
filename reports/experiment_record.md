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

服务器安装 `vllm==0.25.1`，并将 Qwen3.5-4B 从 ModelScope 下载到：

```text
/data02/usr/wangqihao/Demo/test/flashvid/models/Qwen3.5-4B
```

模型配置和两个 safetensors 分片均通过完整性检查。环境、缓存、日志和结果均限定在
项目目录，既有数据集目录保持不变。

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

### 3.1 vLLM 不输出视觉注意力，ADTS 没有可用的重要性分数

**现象：** FlashVID 的 ADTS 需要 Vision Encoder 最后一层逐帧 attention，但 Qwen3.5
在 vLLM 中只返回 vision merger 后的 embeddings。FlashAttention 路径也不会把完整
attention matrix 保存在模型输出中，直接照搬上游代码时没有可传给 ADTS 的分数。

**根因：** 上游 FlashVID 基于能够访问 attention 的模型前向；vLLM 为节省显存，只在
融合 kernel 内计算注意力。与此同时，Qwen3.5 在 attention 后还有 spatial merge，ADTS
需要的 token 粒度和最后传给 LLM 的 token 粒度并不相同。

**解决：** 在最后一个视觉 attention block 的 QKV 投影上注册临时 forward hook，取得
`[sequence, batch, 3, heads, head_dim]` 的 QKV。随后复用同一层的 rotary embedding，按
每个视频的 `grid_thw` 恢复 query/key，并重新计算：

```text
received_attention = sum_query,head softmax(QK^T / sqrt(head_dim))
```

最后按照 Qwen3.5 的 spatial merge unit 求均值，使 attention 分数与 merger 输出 token
一一对应。只重算 Q/K attention，不改变原模型的 vision embeddings。

**验证：** 固定张量测试中，ADTS 选出的索引与上游 greedy rule 完全一致。真实服务中，
如果 QKV 长度、`grid_thw` 或 merge unit 任一不一致，代码会直接抛出长度错误；实际
dummy 权重和 8.8 GB 真实权重均完成视频请求，ratio 从 1.0 调到 0.1 后 prompt tokens
由 11,671 降到 1,756，说明 attention 捕获、merger 粒度转换和 ADTS 输入已经贯通。

### 3.2 ADTS+TSTM 的自然输出数量不等于 vLLM 预先计算的占位 token 数

**现象：** ADTS 的取整、TSTM 的相似度阈值和 DPC-kNN 的聚类数量共同决定最终 token
数，原始算法自然产生的是数据相关长度；但 vLLM 在模型前向前已经根据 retention ratio
生成固定数量的 `<video>` placeholders。两者只要相差 1 个 token，embedding 替换就会
出现长度不匹配。

**根因：** “阈值式合并”只能决定哪些 token 可以合并，不能保证精确预算；逐帧独立
取整还会累积误差。DPC 聚类中心同时承担输出 token 的 anchor，如果中心重复或顺序不
稳定，还会破坏后续位置编码。

**解决：** 将总目标预算拆成 ADTS 和上下文两部分：ADTS 使用 `ceil(target*0.7)`，再用
largest-remainder 方式按帧分配；TSTM 先产生候选，再按剩余预算执行 DPC-kNN。若阈值
合并导致候选不足，则按未使用 token 的 attention 从高到低补齐。最终对 anchor 排序，
并强制检查：

```text
retained_tokens == target_tokens
unique(anchor_indices) == target_tokens
anchor_indices 单调递增
```

**验证：** 在 8 帧×20 token 的固定输入上，`ratio=1.0/0.5/0.25/0.1` 分别严格得到
160/80/40/16 个 token，且所有 anchor 唯一、有序。服务器真实视频四个比例均成功：

| Ratio | Prompt tokens |
|---:|---:|
| 1.00 | 11,671 |
| 0.50 | 6,163 |
| 0.25 | 3,409 |
| 0.10 | 1,756 |

四种服务都没有出现 placeholder/embedding 数量错误。

### 3.3 压缩 embeddings 后，placeholder、时间戳和 M-RoPE 必须同时保持同一索引语义

**现象：** TSTM 会把后续帧 token 合并到前面帧的 anchor。若只把合并后的 embeddings
拼接给 LLM，原始 placeholders 和 M-RoPE 仍按未压缩视频生成；单视频可能表现为位置
错位，多视频还会把不同视频的时间和空间位置串在一起。

**根因：** vLLM 的输入处理阶段先生成 placeholders 和原始 M-RoPE，模型阶段才得到
Vision Encoder 输出。FlashVID 又在模型阶段改变 token 长度，因此必须用同一个 anchor
定义同时描述“保留哪个 embedding”和“该 embedding 在原视频中的位置”。

**解决：** 将每个输出 token 绑定到原视频展平后的唯一 anchor index，生成原长度的
boolean retention mask；再按帧统计保留数量，并把 `retention_mask`、timestamps 和
`video_grid_thw` 交给 Qwen3.5 的 final-video-embedding 路径。M-RoPE 使用压缩后的多模态
embeddings 重新计算。多视频请求按视频分别压缩和重算，最后才按原请求顺序拼接。

**验证：** 纯文本请求保持 17 prompt tokens，图片请求正常返回且不进入视频压缩。单个
视频在 ratio=0.1 时为 1,756 prompt tokens；同一请求放入两个相同视频时为 3,495，正好
满足 `2 × (1756 - 17) + 17 = 3495`，说明两个视频各自的视觉占位长度独立计算，只共享
一次文本 prompt。两个请求均返回 HTTP 200，没有 M-RoPE 或 embedding shape 错误。

### 3.4 ratio=1.0 不能只是“合并后数量相同”，而必须是数值和顺序都不变

**现象：** 如果 ratio=1.0 仍执行 attention 重算、ADTS、聚类和 anchor 重排，即使输出
token 数等于原始数量，也可能因为浮点平均或顺序变化破坏基线，导致所谓“无压缩”服务
不再等价于原生 Qwen3.5。

**根因：** token 数相同不代表 features 相同；任何一次 cluster mean、重新排序或位置
重算都可能引入差异。该问题不能只用长度测试发现。

**解决：** ratio=1.0 时在压缩器最前面直接返回展平后的原始 features 和连续 anchor
`arange(total_tokens)`；模型插件同时关闭 QKV hook 和 multimodal pruning，使该路径不
进入 ADTS/TSTM，也不触发自定义 retention mask。

**验证：** 单元测试对 3×5×4 张量使用 `torch.equal`，确认输出 features 与原张量展平后
逐元素相等，anchor 严格等于 0..14。API 对照中，插件 ratio=1.0 与原生 vLLM 的同一
视频均为 11,671 prompt tokens，并产生完全相同的 64-token greedy 文本内容。

### 3.5 逐帧恢复 attention 会产生大量小 kernel，抵消压缩带来的 prefill 收益

**现象：** 初版 attention 恢复按 frame 循环。若视频有 `T` 帧、每帧 `P` 个 patch、
query chunk 为 128，一次视频会发起约 `T × ceil(P/128)` 组小矩阵计算；帧数增加时 GPU
利用率被 Python 循环和 kernel launch 开销限制。

**根因：** 每帧的 Q/K shape 实际相同，串行循环没有数据依赖，却没有利用 temporal
维度做 batch；直接一次性构造完整 `T×P×P` attention 又会放大显存峰值。

**解决：** 将 Q/K 重排为 `[T, heads, P, head_dim]`，所有帧在 temporal 维并行执行
batched matmul，只在 query 的 P 维保留 chunk。以 32 帧、每帧 256 patch 为例，chunk
128 时，矩阵计算批次从约 64 组降到 2 组；softmax 和累加使用 FP32，最终再映射到
merger token。

**验证：** 32×256×2560 BF16 输入的 GPU 压缩测试中，ratio=0.1 将 8,192 个 token
压到 820 个，压缩模块耗时约 84.0 ms；ratio=0.25 压到 2,048 个，约 153.2 ms。完整
服务测试中，DP8 在 32 请求、16 并发下达到 105.15 output tokens/s，32/32 请求成功；
DP1 为 57.95 output tokens/s，DP8 提升约 81.4%，且没有 OOM。

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
