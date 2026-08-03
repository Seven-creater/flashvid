# EVA-style Agent v1–v4 复现实验记录

## 1. 任务要求

### 1.1 总目标

本实验以同一份 `Qwen3.5-9B` 多模态模型为基础，在长视频选择题上比较：

1. `Direct`：模型直接读取视频并回答；
2. `EVA-style Agent`：参考官方 EVA 的“规划—选帧—观察—反思”流程，模型只读取主动选择的帧；
3. `Hybrid Agent`：先保留 Direct 候选，再用 EVA 官方 `frame_select` 协议取证，只有证据足够时才改判；
4. `Evidence Agent v4`：加入候选盲化检索、CLIP/SGFS 分层定位、密集观察、OCR、结构化证据记忆和双 Judge 裁决。

最终目标不是降低 token，而是验证 Agent 是否能在同一模型、同一固定样本上稳定超过
Direct，并定位效果受限的真实原因。

### 1.2 复现范围与非目标

本实验复用官方
[EfficientVideoAgent](https://github.com/wangruohui/EfficientVideoAgent) 的评测流程和
`select_frame_fallback.py`，包括：

- `<tool_call>` 形式的 `frame_select` 请求；
- `role=tool` 和 `<tool_response>` 观察消息；
- 基于 decord 的指定时间段抽帧及真实时间戳记录；
- 多轮“总结证据—规划新区间—继续观察—最终回答”消息循环；
- 超时、解析失败和最大轮数后的候选回退。

本地作为协议依据的官方仓库固定在 commit
`177493911ca05dfe979dc63c009ffe675258b1de`。

明确没有复现：

- EVA 官方训练权重；
- EVA 的强化学习训练过程；
- EVA 论文全部六个数据集及官方论文指标；
- 任何使用测试集答案训练或微调模型的过程。

因此，本文结论只能称为：

> Qwen3.5-9B Direct 与 training-free EVA-style Agent 在固定 manifest 上的工程对照。

不能称为 EVA 官方权重或论文完整结果复现。

### 1.3 数据、模型与公共约束

| 项目 | 配置 |
|---|---|
| 答题模型 | `Qwen/Qwen3.5-9B` |
| 服务 | vLLM OpenAI-compatible API；常规 DP=8，v4 索引并行阶段临时 DP=7 |
| GPU | 8×NVIDIA A6000 |
| 温度 | `temperature=0` |
| 正式样本 | 每个数据集固定 100 条，`seed=42` |
| 数据集 | LVBench、LSDBench、CG-Bench |
| 并发 | 正式运行默认 32 |
| 长任务 | `setsid + nohup + --resume` |
| Direct | 冻结结果，不因 Agent 版本更新而重跑或覆盖 |
| 数据泄漏 | 不向模型发送答案、`time_range`、`clue_intervals` 或人工 `question_type` |

三个 Direct 基线的原始结果为：

| 数据集 | 原始正确数 | 可解析且与 Agent 共同有效的样本 |
|---|---:|---:|
| LVBench | 45/100 | 94 |
| LSDBench | 31/100 | 83 |
| CG-Bench | 46/100 | 98 |

LVBench 固定 manifest 中有两个源视频已不可获取，因此所有版本均将这两条单独标记为
`data_unavailable`，不把它们当作模型推理错误。

### 1.4 v1–v4 的定义

本文的版本号统一定义如下：

| 版本 | 定义 |
|---|---|
| v1 | Direct 候选 + EVA 取证验证；验证器可较积极改判 |
| v2 | 严格答案解析 + 保守改判 gate + 风险改判二次确认 |
| v3 | 题型路由、多种覆盖/确认策略并行比较；最终选择 `hybrid_v3c` |
| v4 | 候选盲化检索 + CLIP/SGFS 索引 + 密集观察/OCR + 结构化证据记忆 + 双 Judge |

在 v1 之前还运行过不带 Direct 候选的 ordinary Agent，以及低 token 的
`v2a/v2b/v2c`。这些结果只作为转向 Hybrid 的前置证据，不属于本文 v1–v4 编号。

## 2. 实际做法与进展顺序

### 第一步：统一评测层并冻结 Direct

先将 CG-Bench、LVBench 和 LSDBench 转成同一 MCQ 格式，固定 `sample_id`、视频、
问题、可变数量选项和答案。模型侧只读取问题和选项；数据集标注时间和 clue 只在推理
结束后由评分程序读取。

Direct 请求向 Qwen3.5-9B 发送完整视频、问题和所有选项，并要求最终输出
`Answer: X`。生成后的 manifest 和 Direct JSONL 被后续版本共同复用。

完成后的关键能力包括：

- 三个数据集统一入口；
- A–H 可变选项解析；
- 固定 manifest；
- JSONL 逐样本记录；
- 断点续跑；
- 原始回答、预测、token、轮数和延迟统计。

### 第二步：接入官方 EVA 抽帧协议并运行 ordinary Agent

最初的 Agent 不使用 Direct 候选，只让 Qwen3.5-9B 规划 `frame_select`：

```text
问题和选项
  -> 模型请求时间区间
  -> 官方 select_frame_fallback.py 抽帧
  -> tool observation
  -> 模型继续规划或回答
```

普通 Agent v1 在 LVBench 得到 `42/100`，低于 Direct 的 `45/100`。随后按低 token
路线运行：

| 前置版本 | LVBench |
|---|---:|
| Direct | 45/100 |
| ordinary Agent v1 | 42/100 |
| budgeted v2a | 40/100 |
| budgeted v2b | 39/100 |
| budgeted v2c | 37/100 |

增加概览帧、选项排除提示和双区间仍持续下降，说明仅靠“模型自己找帧并直接回答”
无法稳定保留 Direct 对完整视频的已有能力。由此停止继续压 token，转向
“Direct 候选 + 证据验证”的 Hybrid 结构。

### 第三步：v1——Direct 候选 + EVA 反证

Hybrid v1 的流程为：

1. 复用冻结 Direct 结果作为候选；
2. 验证器只接收候选字母，不接收 Direct 原始解释；
3. 要求验证器优先寻找能够推翻候选的证据；
4. 使用官方 `frame_select` 最多多轮取证；
5. 验证器给出最终答案；
6. API、抽帧或答案解析失败时回退候选。

v1 证明了候选回退能避免 ordinary Agent 的大幅退化，但改判过于积极：

| 数据集 | Raw | 共同有效样本：Direct → v1 | 改判 | 改对 | 改错 |
|---|---:|---:|---:|---:|---:|
| LVBench | 46/100 | 45 → 46 | 24 | 9 | 9 |
| LSDBench | 39/100 | 31 → 30 | 21 | 6 | 8 |

LVBench 的改对和改错完全抵消；LSDBench 在共同有效样本上反而比 Direct 少 1 题。
这说明“看过一些帧并给出不同答案”不能等价为可靠反证。

### 第四步：v2——严格解析和保守改判 gate

Hybrid v2 将 Direct candidate 设为默认答案，并增加四项约束：

- 验证器最终答案只接受最后一行 `Answer: X` 或纯字母；
- 长解释中的 `option A`、`A.` 或孤立字母不再作为最终答案；
- 出现 `unclear/likely/appears/seems/not visible/insufficient` 等不确定语义时拒绝改判；
- 动作、事件和时序题若要改判，必须再取一个非重叠区间确认。

结果：

| 数据集 | Raw | 共同有效样本：Direct → v2 | 改判 | 改对 | 改错 |
|---|---:|---:|---:|---:|---:|
| LVBench | 53/100 | 45 → 52 | 13 | 8 | 1 |
| LSDBench | 45/100 | 31 → 33 | 2 | 2 | 0 |

v2 明显降低误改：LVBench harmful changes 从 9 降到 1，LSDBench 从 8 降到 0。
代价是平均延迟上升到 `92.06 s` 和 `96.63 s`，平均视觉 token 分别为
`4954` 和 `3448`。

### 第五步：v3——题型分流与多策略固定比较

v3 不再继续修改同一条提示词，而是预先定义多种互补策略：

- `global_overview`：全时间轴概览后局部 zoom；
- `explicit_question_time`：直接围绕问题中的时间密集观察；
- `ocr_detail`：少帧、高分辨率；
- `action_event`：围绕动作状态变化补充证据；
- `temporal_event`：比较事件前后顺序；
- 改判时执行非重叠证据确认；
- 失败时仍回退冻结候选。

路由只由问题文本生成，数据集 `question_type` 只在离线统计中读取。

预先运行的 v3 变体结果：

| 版本 | LVBench | LSDBench |
|---|---:|---:|
| v3a | 51/100 | 46/100 |
| v3b | 49/100 | 45/100 |
| **v3c** | **53/100** | **51/100** |
| v3d | 49/100 | 45/100 |
| v3e | 47/100 | 45/100 |
| v3f | 49/100 | 45/100 |
| v3g | 52/100 | 48/100 |

按“共同有效样本正确数优先，再减少有害改判和错误”的固定规则选择 `hybrid_v3c`。
其配置为：

- overview 16 帧；
- 最多 6 轮工具取证；
- 每轮一个区间；
- route-specific 帧数和 resize；
- 改判必须再做一次非重叠确认。

v3c 的结果：

| 数据集 | Raw | 共同有效样本：Direct → v3c | 改判 | 改对 | 改错 |
|---|---:|---:|---:|---:|---:|
| LVBench | 53/100 | 45 → 51 | 8 | 6 | 0 |
| LSDBench | 51/100 | 31 → 39 | 9 | 7 | 1 |

相对 v2，v3c 在 LSDBench 的动作题和时序题上各多答对 3 题；LVBench 的时序题多
2 题，但全局概览和 OCR 各少 1 题。v3c 的平均视觉 token 降到 LVBench `1748`、
LSDBench `940`，平均延迟降到 `50.38 s` 和 `55.76 s`。

### 第六步：v4——从提示词优化转向可靠取证

v4 的设计动机是：v3c 的涨分仍可能来自同一 100 条上的版本选择，而且旧 Agent
每个局部区间只有少量帧，未必覆盖完整动作链。因此 v4 新增独立
`evidence_agent` 后端：

```text
冻结 Direct candidate
  -> 候选盲化 Planner
  -> CLIP 每分钟 4 帧粗索引
  -> LSDBench SGFS 保留代表帧
  -> Qwen 语义重排候选事件
  -> 局部 1 FPS / 边界 4 FPS 密集观察
  -> OCR/detail 工具
  -> 带时间戳的结构化 evidence ledger
  -> 候选盲化 Judge
  -> 有冲突时针对性补证
  -> 第二个独立 Judge
  -> 两个 Judge 均支持且存在直接反证时才改判
```

v4 还将运行时样本拆成：

- `ModelSample`：视频、问题、选项、时长和冻结候选；
- `ScoringRecord`：答案、标注时间、clue、人工题型，只供离线评分。

正式结果：

| 数据集 | Direct | 旧最佳 v3c | Evidence v4 |
|---|---:|---:|---:|
| LVBench | 45/100 | 53/100 | 46/100 |
| LSDBench | 31/100 | 51/100 | 36/100 |
| CG-Bench | 46/100 | 未运行 | 46/100 |

共同有效样本配对结果：

| 数据集 | Common | Direct | v4 | Gains | Regressions | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| LVBench | 94 | 45 | 44 | 1 | 2 | 1.0 |
| LSDBench | 83 | 31 | 34 | 3 | 0 | 0.25 |
| CG-Bench | 98 | 46 | 46 | 0 | 0 | 1.0 |

v4 未达到预注册目标，且明显低于 v3c，因此没有设为默认版本。

### 第七步：稳定性、消融、oracle 和结果审计

v4 在三种粗采样偏移和选项排列下运行，不做投票、不挑最好结果：

| 扰动 | LVBench | LSDBench | CG-Bench |
|---|---:|---:|---:|
| offset 0 / perm 101 | 47 | 37 | 45 |
| offset 5 / perm 202 | 45 | 40 | 45 |
| offset 10 / perm 303 | 46 | 38 | 46 |

波动较小，但所有扰动都低于旧最佳，说明 v4 的负结果不是单次随机崩溃。

固定四项消融：

| 消融 | LVBench | LSDBench | CG-Bench |
|---|---:|---:|---:|
| 去掉 blind second Judge | **48** | **42** | 45 |
| 去掉结构化 memory/OCR | 46 | 34 | 46 |
| 去掉 SGFS/语义重排 | 46 | 41 | 46 |
| 密集观察退回 8 帧 | 46 | 39 | 46 |

`no_blind_second_judge` 是三数据集平均最好的消融，说明第二 Judge 对真实改判存在较强
否决效应。`sparse_8` 和去掉 SGFS/重排在 LSDBench 反而高于主版本，也表明增加帧数
没有自动转化为更可靠的证据。

gold-window oracle 直接使用标注时间窗，只用于离线诊断：

| 数据集 | Raw correct | 有效答案 | 有效答案条件准确率 | 无效答案 |
|---|---:|---:|---:|---:|
| LVBench | 44/100 | 70/98 | 62.86% | 28 |
| LSDBench | 40/100 | 53/100 | 75.47% | 47 |
| CG-Bench | 38/100 | 74/100 | 51.35% | 26 |

在能够产出有效答案的样本上，oracle 尤其在 LSDBench 明显更准；但大量密集视觉上下文
无法形成有效结构化答案。这将问题进一步定位到“证据压缩和裁决可靠性”，而不是单纯
缺少目标帧。

## 3. 出现的问题与解决过程

### 3.1 不带 Direct 候选的 Agent 持续低于原模型

**现象：** LVBench 上 Direct 为 `45/100`，ordinary Agent 为 `42/100`；
`v2a/v2b/v2c` 依次降到 `40/39/37`。增加概览帧和双区间没有恢复准确率。

**根因：** Direct 能通过完整视频建立全局语义；ordinary Agent 必须在极少帧上同时完成
定位和推理。一旦首轮概览漏掉关键动作，后续区间会在错误假设上继续缩放。没有 Direct
候选时，抽帧失败或证据不足也没有可靠答案可以回退。

**解决：** 将 Direct 输出从“对照结果”升级为 Hybrid 内部的候选假设。验证器只看到
候选字母，不看到 Direct 推理文本；所有 API、抽帧和解析失败都回退候选。

**验证：** Hybrid v1 的 raw accuracy 恢复到 LVBench `46/100`、LSDBench `39/100`，
没有再出现 budgeted Agent 的整体坍塌。但 v1 的改判审计同时暴露了新的误改问题。

### 3.2 v1 的改判过于积极，正确候选被有限帧推翻

**现象：** v1 在 LVBench 改判 24 次，其中改对 9 次、改错 9 次；LSDBench 改判
21 次，改对 6 次、改错 8 次。共同有效样本上 LSDBench 从 Direct 的 31 题降到 30 题。

**根因：** 验证器可以从任意长解释中抽取选项字母，且只需一次有限帧观察即可改变候选。
模型经常用未看到的动作补全动作链，或把 `likely/appears/seems` 这类不确定结论当作
视觉反证。

**解决：** v2 同时收紧答案语法和改判条件：只接受严格结尾答案；候选默认保留；不确定
语义拒绝改判；动作、事件和时序题必须执行一次额外非重叠确认。

**验证：** LVBench harmful changes 从 `9` 降到 `1`，共同有效正确数从 `46` 增至
`52`；LSDBench harmful changes 从 `8` 降到 `0`，共同有效正确数从 `30` 增至
`33`。该结果证明提升主要来自减少错误改判，而不是单纯增加工具轮数。

### 3.3 单一提示词在不同题型和数据集上的收益不一致

**现象：** v2 在 LVBench 达到 `53/100`，但 LSDBench 只有 `45/100`；同一策略对
显式时间题、全局汇总、OCR、动作链和时序题的帧密度要求不同。继续统一增加分辨率会将
平均延迟推到约 96 秒，却没有等比例改善 LSDBench。

**根因：** 全局题需要时间轴覆盖，OCR 需要少帧高分辨率，动作/顺序题需要连续局部帧。
统一 overview 和确认策略会在不同题型之间制造冲突。数据集的人工 `question_type`
又属于私有标注，不能作为运行时路由输入。

**解决：** v3 只根据问题文本路由，并预先冻结 v3a–v3g 的帧数、resize、轮数和确认
方式。以共同有效正确数、有害改判和错误数选择 v3c，不在单个坏例上临时换策略。

**验证：** v3c 在 LSDBench 动作题和时序题上分别比 v2 多答对 3 题，raw accuracy
从 `45/100` 增至 `51/100`；平均视觉 token 从 `3448` 降到 `940`，平均延迟从
`96.63 s` 降到 `55.76 s`。LVBench raw score 维持 `53/100`。

需要注意：v3c 是从 7 个版本中在同一 100 条上选出的版本，这一结果存在明确的版本选择
偏差，只能作为固定样本上的工程改善，不能当作独立盲测结论。

### 3.4 v4 增加密集取证后反而退化

**现象：** v4 使用更复杂的检索和更多帧后，LVBench/LSDBench/CG-Bench 仅为
`46/36/46`。模型未形成有效证据答案的内部流程失败分别有 `29/76/33` 条。
错误 Direct 候选被原样保留 `48/48/52` 条，真正改对仅 `1/3/0` 条。

**根因：** v4 解决了“取更多目标帧”，但没有解决“如何把大量帧稳定压成可裁决事实”。
LSDBench 平均有 71.76 个目标区间帧，仍有 76% 样本未形成有效 evidence answer。
长视觉上下文被拆成多个观察块后，动作顺序、计数和跨事件状态在 ledger 中丢失；随后
严格双 Judge 又进一步否决候选变化。

离线诊断也表明帧命中和准确率不呈稳定单调关系：

| 数据集 | 命中目标窗准确率 | 未命中准确率 | 至少 30 个目标帧准确率 |
|---|---:|---:|---:|
| LVBench | 42.11% | 53.66% | 45.45% |
| LSDBench | 37.21% | 28.57% | 43.64% |
| CG-Bench | 51.79% | 38.64% | 50.00% |

**解决：** v4 没有继续通过提示词调参掩盖失败，而是固定运行 gold-window oracle、三次
扰动和四项消融，将定位失败和观察/裁决失败分开。最终不启用 v4，保留 v3c 为当前最佳
工程版本。

**验证：** gold-window oracle 在有效答案上的条件准确率为
`62.86%/75.47%/51.35%`，说明正确窗口确实有用；但无效答案达到 `28/47/26` 条。
去掉第二 Judge 后 v4 提升到 `48/42/45`，仍未超过 v3c。证据共同支持：
下一步应先做窗口级原子事实压缩和小上下文裁决，而不是继续增加帧数。

### 3.5 标注隔断既要防真实泄漏，也要避免把模型输出误报为泄漏

**现象：** 稳定性运行中，一条样本被标记为 `annotation_leak_detected`。检查请求后发现
私有标注没有进入模型；是模型自己生成的 JSON-like 文本包含 `correct_answer` 字样，
旧检查器扫描整段字符串后误报。

**根因：** 仅用正则禁止关键词无法区分：

- 数据集对象中的真实私有字段；
- 普通问题文本；
- 模型自己生成、随后被下一轮引用的证据文本。

**解决：**

1. 使用类型隔断：模型代码只接受不含 `answer/metadata` 的 `ModelSample`；
2. 真实答案和时间标注存入 `ScoringRecord`，推理结束后再 join；
3. 请求审计递归检查真实结构键和秘密 sentinel；
4. 不再把模型生成文本中的同名词当作真实标注字段；
5. 缺失视频单独记录为 `data_unavailable`。

**验证：**

- 最终测试：本地与服务器均为 `54 passed`；
- 28 个结果文件，共 2710 条记录；
- 重复 `sample_id=0`；
- annotation leak failures `=0`；
- 进程级 API/抽帧异常 `=0`；
- 唯一数据异常是 LVBench 两个不可用源视频，所有运行保持一致。

## 4. 简洁实验报告

### 4.1 主要结果

v1–v4 的原始正确数：

| 版本 | LVBench | LSDBench | CG-Bench |
|---|---:|---:|---:|
| Direct | 45 | 31 | 46 |
| Hybrid v1 | 46 | 39 | 未运行 |
| Hybrid v2 | 53 | 45 | 未运行 |
| **Hybrid v3c** | **53** | **51** | 未运行 |
| Evidence v4 | 46 | 36 | 46 |

共同有效样本上的严格比较：

| 版本 | LVBench（94条） | 相对 Direct | LSDBench（83条） | 相对 Direct |
|---|---:|---:|---:|---:|
| Direct | 45 | — | 31 | — |
| v1 | 46 | +1 | 30 | -1 |
| v2 | 52 | +7 | 33 | +2 |
| **v3c** | **51** | **+6** | **39** | **+8** |
| v4 | 44 | -1 | 34 | +3 |

成本：

| 版本 | 数据集 | 平均轮数 | 平均视觉 token | 平均总 token | 平均延迟 |
|---|---|---:|---:|---:|---:|
| v1 | LVBench | 3.52 | 4002 | 29189 | 59.43 s |
| v1 | LSDBench | 3.33 | 2874 | 25429 | 52.99 s |
| v2 | LVBench | 3.61 | 4954 | 32282 | 92.06 s |
| v2 | LSDBench | 3.88 | 3448 | 28576 | 96.63 s |
| **v3c** | LVBench | 3.78 | 1748 | 27425 | 50.38 s |
| **v3c** | LSDBench | 3.94 | 940 | 25371 | 55.76 s |
| v4 | LVBench | 7.64 | 16772 | 20979 | 62.77 s |
| v4 | LSDBench | 9.19 | 15035 | 25928 | 80.40 s |
| v4 | CG-Bench | 7.63 | 44383 | 41876 | 85.69 s |

注意：视觉 token 是抽帧工具的估算量，总 token 来自 API usage，两者不是简单相加关系。

### 4.2 结论

- ordinary EVA-style Agent 不能稳定超过完整视频 Direct，低 token 版本反而持续下降。
- Direct candidate 是有效保底，但 v1 证明没有可靠 gate 时“改对”和“改错”会互相抵消。
- v2 的主要收益来自严格解析和减少误改，不是单纯增加取证轮数。
- v3c 是当前测试过的最佳平衡版本：LVBench raw `53/100`、LSDBench `51/100`，
  同时比 v2 使用更少视觉 token 和更低延迟。
- v4 是明确的负结果。更多帧、更复杂检索和双 Judge 没有自动带来提升；主要瓶颈是
  密集视觉证据无法被稳定压缩成短、可校验、可裁决的事实。
- v4 不应替代 v3c；其价值在于通过消融和 oracle 把问题从“继续调提示词”定位到
  “证据表示与裁决可靠性”。
- 全部结论均基于反复使用过的固定 100 条 manifest，属于工程改善，不是新的统计盲测。

### 4.3 尚未完成

1. 没有下载或运行 EVA 官方权重，也没有复现强化学习训练。
2. v1–v3 没有在 CG-Bench 上运行；CG-Bench 只有 Direct 和 v4 的同 manifest 对照。
3. v3c 是从 7 个版本中在同一测试集上选出，尚需全新隐藏集验证泛化。
4. 尚未实现“每个局部窗口独立生成原子事实，再由小上下文 Judge 裁决”的下一代方案。
5. 尚未对选项 logprob 做校准，当前 gate 仍以结构化证据和规则为主。
6. LVBench 有两个源视频不可用，raw denominator 保持 100，但共同有效比较会排除无效样本。
7. 没有运行 EVA 论文中的 VideoMME、LongVideoBench、MLVU 和 VideoHolmes。

### 4.4 证据与复现入口

主要证据：

- v1–v4 原始 JSONL、汇总与审计文件保留在服务器旧实验目录中，只读不覆盖。
- v4 源码、诊断脚本和小型报告已在 2026-08-03 清理时移出活动仓库，保存在仓库外归档；本记录保留其负结果结论用于审计。
- 新实验分支不导入、不运行 v4，也不以 v4 为后续实现基础。

核心代码：

- [Hybrid v1–v3 控制器](../src/flashvid_eval/runner.py)
- [统一评测入口](../scripts/evaluate_mcq.py)
- [官方 EVA 抽帧代码](../third_party/EfficientVideoAgent/select_frame_fallback.py)

典型复现命令：

```bash
# 当前最佳 Hybrid v3c
python scripts/evaluate_mcq.py \
  --dataset lsdbench \
  --backend hybrid \
  --agent-version hybrid_v3c \
  --annotations /data02/pretrained_model/cvr_learn/cvr_data/07_lsdbench/test.json \
  --video-root /data02/pretrained_model/cvr_learn/cvr_data/07_lsdbench/videos_partial \
  --base-url http://127.0.0.1:8001/v1 \
  --model Qwen3.5-9B \
  --manifest results/eval/lsdbench_manifest_42_100.jsonl \
  --candidate-results results/eval/lsdbench_direct.jsonl \
  --frame-root /dev/shm/flashvid_eval_frames \
  --output-dir results/eval/optimized_agent/hybrid_v3c \
  --max-turns 6 \
  --max-call-visual-tokens 12000 \
  --max-total-visual-tokens 24000 \
  --concurrency 32 \
  --resume

```
