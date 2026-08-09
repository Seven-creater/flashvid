# Perception-Memory EVA SFT 运行手册

这份手册只规定安全执行顺序，不代替各阶段脚本。任何门槛失败都必须停在当前阶段；不得通过修改 Test300、正式 prompt、冻结候选或阈值继续运行。

## 1. 仅在服务器执行预检

本地电脑不得下载模型、数据或训练依赖。服务器若确实缺少依赖，只能使用已批准的国内镜像：

```bash
cd /data02/usr/wangqihao/Demo/test/qwen_agent_search_sft
export HF_ENDPOINT=https://hf-mirror.com
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PYTHONPATH="$PWD:$PWD/src"

python scripts/validate_perception_memory_config.py \
  --config configs/experiments/perception_memory_eva_sft.json
```

随后用现有切分审计器核验文件哈希、每库 `200/50/100` 条和视频级零交叉。该命令只允许创建与既有内容完全一致的冻结 Train600；若源文件、哈希或隔离关系发生变化会直接失败：

```bash
python scripts/freeze_qwen_train600.py \
  --config configs/experiments/perception_memory_eva_sft.json \
  --output results/eval/perception_memory_eva_sft/frozen/train600.jsonl \
  --metadata results/eval/perception_memory_eva_sft/frozen/train600.metadata.json
```

核验 `train600.metadata.json` 的 `video_isolation` 为 `passed`，输出 SHA-256 为 `3995454d973aeb6efe6887e821cd5197e0f17a5b9cc32d7c719e6583b487e6c0` 后才能继续。

## 2. 当前 SFT 配对审计是硬门槛

先在服务器定位本次未训练与本次 SFT 的三库 Test300 JSONL。不要用旧 v4 结果替代。把下面六个变量指向真实文件后执行：

```bash
: "${UNTRAINED_LV:?}" "${UNTRAINED_LSD:?}" "${UNTRAINED_CG:?}"
: "${SFT_LV:?}" "${SFT_LSD:?}" "${SFT_CG:?}"

python scripts/audit_perception_memory_badcases.py \
  --untrained "$UNTRAINED_LV" "$UNTRAINED_LSD" "$UNTRAINED_CG" \
  --sft "$SFT_LV" "$SFT_LSD" "$SFT_CG" \
  --manifest \
    /data02/usr/wangqihao/Demo/test/flashvid/results/eval/flashvid_budget_v1/frozen/manifests/lvbench_manifest_42_100.jsonl \
    /data02/usr/wangqihao/Demo/test/flashvid/results/eval/flashvid_budget_v1/frozen/manifests/lsdbench_manifest_42_100.jsonl \
    /data02/usr/wangqihao/Demo/test/flashvid/results/eval/flashvid_budget_v1/frozen/manifests/cgbench_manifest_42_100.jsonl \
  --output-dir results/eval/perception_memory_eva_sft/diagnostics/current_sft_badcases
```

必须人工复核 `summary.json`、`paired_badcases.jsonl` 和请求轨迹，并完成配置中的七类失败归因与四级漏斗。硬门槛是总计300条、每库100条、无重复ID。SSH不可达、文件缺失或ID不一致时停止，不能先生成新轨迹。

## 3. 运行时与 SFT 数据门槛

在开始批量任务前先运行三库 smoke 与相关测试。Controller 请求媒体数必须为0；Perception每轮只接收新帧，不能接收候选、答案、时间标注、clue或人工题型。

轨迹与前缀构建完成后，先生成机器可读 gate 报告，至少满足：

- 稳定正确题不少于360条，每库不少于100条；
- 错误候选被视觉证据改正不少于90条，每库不少于20条；
- 只有 evidence-only Judge 3/3 正确的完整前缀可监督 `stop`；
- 删除最后观察后不再稳定正确的前缀必须作为 `continue + frame_select` 样本保留；
- annotation leak、candidate rerun、重复ID均为0。

任一条件失败时不启动训练。通过后使用已有 `ms-swift==4.4.2` 和本地 Qwen3.5-9B 权重；不在本机下载，也不新增模型。

### 3.1 从缓存帧生成逐步视觉证据

先设置固定路径。`OLD_RAW` 必须是 Train600 上旧 Fast Hybrid Teacher 的原始轨迹，不得指向 Test300：

```bash
PM_ROOT="$PWD/results/eval/perception_memory_eva_sft"
DIAG="$PM_ROOT/diagnostics/current_sft_badcases/summary.json"
TRAIN600="$PWD/results/eval/qwen_agent_search/frozen/train600.jsonl"
TRAIN600_SHA=3995454d973aeb6efe6887e821cd5197e0f17a5b9cc32d7c719e6583b487e6c0
OLD_RAW="${OLD_RAW:?point to a Train600 Fast Hybrid raw trajectory JSONL}"
OBS_BASE="$PM_ROOT/trajectories/observation_states/base.jsonl"
PREFIX_BASE="$PM_ROOT/trajectories/prefixes/base_judgments.jsonl"
mkdir -p "$PM_ROOT/logs" "$(dirname "$OBS_BASE")" "$(dirname "$PREFIX_BASE")"
```

重放只复用已经存在的帧和时间戳；缺帧会失败，绝不重新抽帧或静默换采样方式：

```bash
setsid nohup python scripts/replay_perception_memory_trajectories.py \
  --input "$OLD_RAW" \
  --output "$OBS_BASE" \
  --audit-summary "$DIAG" \
  --base-url http://127.0.0.1:8200/v1 \
  --model Qwen3.5-9B \
  --concurrency 32 \
  --resume \
  > "$PM_ROOT/logs/replay_base.log" 2>&1 < /dev/null &
```

任务结束后运行三个 candidate-blind、text-only Judge。若只有部分 seed 失败，用第二条命令只重跑失败 seed，成功 seed 保持不变：

```bash
setsid nohup python scripts/judge_perception_memory_prefixes.py \
  --trajectories "$OBS_BASE" \
  --output "$PREFIX_BASE" \
  --base-url http://127.0.0.1:8200/v1 \
  --model Qwen3.5-9B \
  --concurrency 32 \
  --resume \
  > "$PM_ROOT/logs/prefix_base.log" 2>&1 < /dev/null &

# 仅在日志报告 complete_with_failures 时执行：
python scripts/judge_perception_memory_prefixes.py \
  --trajectories "$OBS_BASE" --output "$PREFIX_BASE" \
  --base-url http://127.0.0.1:8200/v1 --model Qwen3.5-9B \
  --concurrency 32 --resume --retry-errors
```

### 3.2 离线关联 Train600 标签并只救援无稳定轨迹样本

真实答案只在上面所有模型请求完成后由下列离线步骤读取。脚本会校验 Train600 的 SHA、600条和每库200条，并生成三库 `*_no_stable.jsonl`：

```bash
python scripts/select_perception_memory_trajectories.py \
  --trajectories "$OBS_BASE" \
  --prefix-judgments "$PREFIX_BASE" \
  --answers "$TRAIN600" \
  --expected-answers-sha256 "$TRAIN600_SHA" \
  --labeled-output "$PM_ROOT/trajectories/selected/base_labeled.jsonl" \
  --selected-output "$PM_ROOT/trajectories/selected/base_selected.jsonl" \
  --summary "$PM_ROOT/trajectories/selected/base_summary.json" \
  --rescue-manifest-dir "$PM_ROOT/trajectories/rescue_manifests"
```

只对这些 no-stable 私有训练 manifest 依次运行四个预注册变体：

```text
rescue_global32
rescue_global64
rescue_first_half64
rescue_second_half64
```

每个变体的第一步只根据视频总时长做固定覆盖，不读取答案、`time_range`、clue或人工题型；后续恢复正常 Controller。调用统一评测入口时必须同时提供：对应库原始 `ANNOTATIONS`、no-stable `MANIFEST`、完整冻结 Train200 `CANDIDATES`、其真实 SHA，以及下面这些固定参数：

```bash
python scripts/evaluate_mcq.py \
  --dataset "$DATASET" --backend perception_memory_eva \
  --agent-version perception_memory_v1 \
  --annotations "$ANNOTATIONS" --video-root "$VIDEO_ROOT" \
  --manifest "$MANIFEST" --expected-manifest-sha256 "$MANIFEST_SHA" \
  --candidate-results "$CANDIDATES" \
  --diagnostics-gate-summary "$DIAG" \
  --defer-scoring --train600-manifest-sha256 "$TRAIN600_SHA" \
  --trajectory-schedule-id "pm-rescue-${DATASET}-v1" \
  --trajectory-variant-id "$VARIANT" \
  --experiment-config-sha256 "$EXPERIMENT_CONFIG_SHA" \
  --model-artifact-sha256 "$MODEL_ARTIFACT_SHA" \
  --model Qwen3.5-9B --qwen-protocol no_think \
  --base-url http://127.0.0.1:8200/v1 \
  --max-turns 6 --max-frames-per-call 128 --concurrency 32 \
  --output-dir "$PM_ROOT/trajectories/rescue/$VARIANT/$DATASET" \
  --resume
```

四个变体完成后，把 base 与 rescue 的所有轨迹/前缀结果一起再次传给 `select_perception_memory_trajectories.py --overwrite`。随后构建训练集；四个数量门槛不能通过命令行调低：

```bash
python scripts/build_perception_memory_sft.py \
  --selected "$PM_ROOT/trajectories/selected/final_selected.jsonl" \
  --output "$PM_ROOT/sft_data/perception_memory_sft.jsonl" \
  --summary "$PM_ROOT/sft_data/summary.json"
```

## 4. 后台训练与 Dev 放行

所有长任务都采用同一安全形式；`COMMAND` 必须包含该阶段的 `--resume` 和冻结配置指纹：

```bash
mkdir -p logs
setsid nohup bash -lc 'COMMAND --resume' \
  > logs/perception_memory_STAGE.log 2>&1 < /dev/null &
```

不要启动巨型自动驾驶脚本。每个阶段完成或异常退出后再审计状态文件、日志、行数、重复ID、泄漏与GPU归属。

先执行绑定到正式输出目录的一步 smoke；只有 smoke 报告通过后才能启动三轮正式训练：

```bash
TRAIN_DATA="$PM_ROOT/sft_data/perception_memory_sft.jsonl"
SMOKE_DIR="$PM_ROOT/checkpoints/smoke"
FORMAL_DIR="$PM_ROOT/checkpoints/formal"

setsid nohup bash scripts/train_qwen_agent_9b_lora.sh \
  --train-data "$TRAIN_DATA" --output-dir "$SMOKE_DIR" \
  --formal-output-dir "$FORMAL_DIR" --smoke --load-weights-preflight \
  > "$PM_ROOT/logs/sft_smoke.log" 2>&1 < /dev/null &

setsid nohup bash scripts/train_qwen_agent_9b_lora.sh \
  --train-data "$TRAIN_DATA" --output-dir "$FORMAL_DIR" \
  --smoke-report "$SMOKE_DIR/preflight/training_update.json" --resume \
  > "$PM_ROOT/logs/sft_formal.log" 2>&1 < /dev/null &
```

启动前仍由训练脚本检查8卡是否每卡至少有42 GiB空闲；不满足时按脚本规则使用4–7卡，不得停止他人进程。

每个epoch先只跑 Dev seed42。只有初筛通过的checkpoint才补 seed17/73。三次平均必须相对同运行时未训练基线增加至少3/150、三库分别不下降、总Token和视觉Token均不超过70%，并满足失败率和泄漏门槛。只冻结一个Dev赢家。

最终 Dev gate 配置必须精确包含 seeds `17/42/73`、三库固定 manifest SHA、baseline和候选的三库结果路径。运行：

```bash
python scripts/gate_perception_memory.py \
  --config "$PM_ROOT/dev_eval/gate_config.json" \
  --output "$PM_ROOT/dev_eval/gate_report.json"
```

Gate只接受 `candidate_cost_complete=true` 且两种 `end_to_end_*_tokens_complete=true` 的样本，绝不会用Agent-only Token冒充端到端成本。

## 5. Test300 只能运行一次

只有唯一Dev赢家的 gate 报告明确为 `passed` 才能启动固定Test300。运行前再次计算三个manifest SHA-256，并与配置逐字匹配。Test300不得用于换checkpoint、改prompt或重训。

最终验收是：相对同运行时未训练Agent至少多6/300、总正确数不少于157/300、三库分别不少于45/63/43且不低于各自同运行时基线，同时平均总Token和视觉Token均不超过70%。不达标就如实报告，不追加RL或继续用测试badcase调参。

Test gate 配置只允许 seed42，并必须写入已通过的 Dev gate报告路径和SHA、同一个唯一 `selected_method_id` 及对应模型artifact。最终运行：

```bash
python scripts/gate_perception_memory.py \
  --config "$PM_ROOT/final_test/gate_config.json" \
  --output "$PM_ROOT/final_test/gate_report.json"
```
