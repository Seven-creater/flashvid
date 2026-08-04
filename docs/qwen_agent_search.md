# Qwen-only 长视频 Agent 运行手册

本实验只允许 Qwen3.5-4B、Qwen3.5-9B 和无语义抽帧工具。固定 300 条仅在 Dev 搜索、轨迹筛选和 checkpoint 门控全部冻结后运行；它是“固定工程测试集”，不是统计盲测集。

## 1. 初始化与硬约束

```bash
export PROJECT_DIR=/data02/usr/wangqihao/Demo/test/qwen_agent_search
export OLD_ENV=/data02/usr/wangqihao/Demo/test/flashvid/.venv
export PYTHON_BIN=$OLD_ENV/bin/python
export VLLM_BIN=$OLD_ENV/bin/vllm
export PYTHONPATH=$PROJECT_DIR/src
cd "$PROJECT_DIR"

$PYTHON_BIN scripts/freeze_qwen_train600.py \
  --config configs/experiments/qwen_agent_search.json \
  --output results/eval/qwen_agent_search/frozen/train600.jsonl \
  --metadata results/eval/qwen_agent_search/frozen/train600.metadata.json

$PYTHON_BIN scripts/preflight_qwen_agent_search.py \
  --config configs/experiments/qwen_agent_search.json \
  --source-root "$PROJECT_DIR" \
  --output results/eval/qwen_agent_search/frozen/preflight.json
```

Train600 SHA 必须是 `3995454d973aeb6efe6887e821cd5197e0f17a5b9cc32d7c719e6583b487e6c0`。启动前用 `nvidia-smi` 确认 GPU 4–7 空闲；不得触碰 GPU 0–3 或非本项目 PID。所有长任务使用 `setsid + nohup`，不创建定时监控。

## 2. 服务、协议与 Direct

启动 9B 服务：

```bash
CUDA_DEVICES=4,5,6,7 VLLM_BIN=$VLLM_BIN \
  bash scripts/launch_qwen_agent_service.sh \
  /data02/usr/wangqihao/Demo/test/eva_baseline/models/Qwen3.5-9B \
  Qwen3.5-9B 8200 4
```

先运行 q9 的三库各10条协议 smoke。smoke 只检查服务、实际抽帧、严格答案解析、
thinking 分离、截断和 Token 记账，不据此调整准确率：

```bash
PYTHON_BIN=$PYTHON_BIN bash scripts/launch_qwen_agent_phase.sh \
  configs/experiments/qwen_agent_search.json protocol_smoke q9
```

当前冻结的 thinking 协议从首轮即统一使用 `max_tokens=32768`，不得混入旧的
8192→32768 重试结果。no-thinking 和 thinking 的最终答案请求都必须使用按本题实际
选项字母生成的 JSON Schema；只有可能返回工具调用的 Agent 规划轮不得套用答案
Schema。vLLM 的 Qwen3 reasoning parser 会保留独立 reasoning，只约束正式 content。
smoke 完整通过后，
再运行 q9 协议审计：

```bash
PYTHON_BIN=$PYTHON_BIN bash scripts/launch_qwen_agent_phase.sh \
  configs/experiments/qwen_agent_search.json protocol_audit q9
```

若 smoke 被单条瞬时 API/网络错误拦下，先检查该行和服务日志，再用同一命令追加
`--retry-errors`。恢复标志只补跑错误行，不改变冻结 run plan，也不会重复已完成样本。
冻结 run-plan 仍记录与旧结果一致的 3600 秒上限；正式启动器通过不进入语义指纹的
`--request-timeout 80` 设置基础设施恢复上限。超过该时间会明确记录失败并继续，且阶段
启动器也会在结果文件连续 90 秒没有增长时终止当前可续跑进程。该恢复参数不会
静默改采样协议或降低帧数。

任务结束后用 `bash scripts/stop_qwen_agent.sh 8200` 停止本项目服务，启动 4B，
依次以 `q4` 运行 `protocol_smoke` 和 `protocol_audit`。两者完成后冻结协议：

```bash
$PYTHON_BIN scripts/select_qwen_protocol_dev.py \
  --config configs/experiments/qwen_agent_search.json \
  --run-plan results/eval/qwen_agent_search/run_plans/protocol_audit_q9.json \
  --run-plan results/eval/qwen_agent_search/run_plans/protocol_audit_q4.json \
  --output results/eval/qwen_agent_search/frozen/protocol_selection.json
```

从该文件读取 `$Q9_PROTOCOL` 和 `$Q4_PROTOCOL`。分别在正确服务上运行：

```bash
PYTHON_BIN=$PYTHON_BIN bash scripts/launch_qwen_agent_phase.sh \
  configs/experiments/qwen_agent_search.json blind_diagnostics q9 \
  --protocol "$Q9_PROTOCOL"
PYTHON_BIN=$PYTHON_BIN bash scripts/launch_qwen_agent_phase.sh \
  configs/experiments/qwen_agent_search.json direct_dev q9 \
  --protocol "$Q9_PROTOCOL"
```

q4 将 `q9/Q9` 替换为 `q4/Q4`。错误视频诊断前先冻结三库、三个 seed 的同库同
时长桶映射；已有文件只有逐字一致才允许复用，不会覆盖不同产物：

```bash
$PYTHON_BIN scripts/freeze_qwen_mismatched_videos.py \
  --config configs/experiments/qwen_agent_search.json
```

`question_choices` 是真正无视频；`choices_only` 连问题也删除；二者视觉 Token 必须为 0。

## 3. A0–A4 Dev 搜索

生产配置预注册 6 个互补搜索点。`--search-variant all` 一次顺序执行这 6 点，避免原 54 点中大量实际执行完全相同的组合。

9B 服务运行时，对以下框架逐个执行；必须等当前后台 PID 退出后再启动下一个：

```bash
PYTHON_BIN=$PYTHON_BIN bash scripts/launch_qwen_agent_phase.sh \
  configs/experiments/qwen_agent_search.json agent_dev q9 \
  --protocol "$Q9_PROTOCOL" \
  --framework a0_eva_clean \
  --search-variant all
```

框架顺序固定为：

1. `a0_eva_clean`
2. `a1_storyboard_zoom`
3. `a2_multi_clue_memory`
4. `a3_hierarchical_search`
5. `a4_independent_arbitration`

全部完成后先冻结唯一的 run-plan allowlist。不要对运行目录做 glob；恢复计划和原计划
可能包含相同逻辑任务，显式索引会在写入前拒绝这种重复。Q4 thinking 在 smoke 被拒绝的
证据也放进同一个索引，避免选择时漏传：

```bash
$PYTHON_BIN scripts/freeze_qwen_selection_plan_index.py \
  --config configs/experiments/qwen_agent_search.json \
  --run-plan results/eval/qwen_agent_search/run_plans/protocol_audit_q9.json \
  --run-plan results/eval/qwen_agent_search/run_plans/protocol_audit_q4_no_think.json \
  --run-plan results/eval/qwen_agent_search/run_plans/direct_dev_q9_no_think.json \
  --run-plan results/eval/qwen_agent_search/run_plans/direct_dev_q4_no_think.json \
  --run-plan results/eval/qwen_agent_search/run_plans/agent_dev_q9_no_think_a0_eva_clean_all.json \
  --run-plan results/eval/qwen_agent_search/run_plans/agent_dev_q9_no_think_a1_storyboard_zoom_all.json \
  --run-plan results/eval/qwen_agent_search/run_plans/agent_dev_q9_no_think_a2_multi_clue_memory_all.json \
  --run-plan results/eval/qwen_agent_search/run_plans/agent_dev_q9_no_think_a3_hierarchical_search_all.json \
  --run-plan results/eval/qwen_agent_search/run_plans/agent_dev_q9_no_think_a4_independent_arbitration_all.json \
  --reject-q4-think-from-smoke results/eval/qwen_agent_search/run_plans/protocol_smoke_q4.json \
  --output results/eval/qwen_agent_search/frozen/q9_dev_selection_plan_index.json
```

随后仅从该索引冻结 q9 winner：

```bash
$PYTHON_BIN scripts/select_qwen_agent_dev_winner.py \
  --config configs/experiments/qwen_agent_search.json \
  --selection-plan-index results/eval/qwen_agent_search/frozen/q9_dev_selection_plan_index.json \
  --teacher-model-key q9 \
  --summary-output results/eval/qwen_agent_search/frozen/q9_dev_selection.json \
  --winner-output results/eval/qwen_agent_search/frozen/q9_agent_winner.json
```

从 `q9_dev_selection.json` 读取每个 `accepted=true` 阶段的 `stage` 与
`variant_id`，在 4B 服务上用 `agent_dev q4 --framework <stage>
--search-variant <variant_id>` 运行完全相同的三 seed 配置。4B 结果只作模型规模对照，
不反向改变 9B Teacher winner，也不在固定 300 条上挑配置。

选择器强制 A0→A4 顺序、三个 seed、平均至少多 2 题且至少两个 seed 获胜。A5 禁用；没有框架通过 Direct 门槛时不会生成 winner。

## 4. Teacher、轨迹与 SFT

先在 Dev150 重放冻结 Teacher：

```bash
PYTHON_BIN=$PYTHON_BIN bash scripts/launch_qwen_agent_phase.sh \
  configs/experiments/qwen_agent_search.json teacher_dev q9 \
  --frozen-winner-config results/eval/qwen_agent_search/frozen/q9_agent_winner.json
```

再在 Train600 生成每题 12 条不同“实际取证日程”的轨迹：

```bash
PYTHON_BIN=$PYTHON_BIN bash scripts/launch_qwen_agent_phase.sh \
  configs/experiments/qwen_agent_search.json trajectory q9 \
  --frozen-winner-config results/eval/qwen_agent_search/frozen/q9_agent_winner.json
```

每条轨迹同时绑定单库 Train200 SHA 和 Train600 SHA，且由 seed 17/42/73 三个独立 Judge 复核。`build_qwen_agent_sft.py --phase counterfactuals` 对所有稳定正确基础族生成 75%/50%/25% 帧数与前缀删除规格；`run_qwen_counterfactuals.py` 仅重放冻结区间。最终 `--phase select` 只保留 3/3 正确且总 Token 最低的代表轨迹。

基础 12 条轨迹完成后，先用 `collect_qwen_sft_inputs.py` 生成不含反事实的 base bundle。`freeze_qwen_rescue_inputs.py` 离线连接 Train600 标签，只把 12 条均无稳定正确轨迹的 sample ID 写入三个冻结子 manifest，并固定一条 128 帧概览、A3 密集证据分支、uniform128 独立 Direct 与仲裁组成的 `rescue_a4_dense_v1`。模型请求仍由 `ModelSample` 隔断标签。随后用 `launch_qwen_rescue.sh` 运行；救援行的 `variant_id=rescue`，因此不会被误计为第 13 条基础 schedule。最终 bundle 通过 `collect_qwen_sft_inputs.py --rescue-index ...` 同时纳入这些行，稳定正确的救援轨迹也会进入删帧反事实阶段。

```bash
$PYTHON_BIN scripts/freeze_qwen_rescue_inputs.py \
  --config configs/experiments/qwen_agent_search.json \
  --base-bundle results/eval/qwen_agent_search/trajectories/base_bundle.json \
  --frozen-winner results/eval/qwen_agent_search/frozen/q9_agent_winner.json \
  --output-dir results/eval/qwen_agent_search/trajectories/rescue/frozen

PYTHON_BIN=$PYTHON_BIN QWEN_STALL_TIMEOUT_S=90 bash scripts/launch_qwen_rescue.sh \
  results/eval/qwen_agent_search/trajectories/rescue/frozen/rescue_index.json
```

不要手工拼接轨迹路径或哈希。先把36个基础轨迹任务冻结成一个自校验输入 bundle：

```bash
CONFIG_SHA=$($PYTHON_BIN -c 'import json,sys; from flashvid_eval.qwen_sft import canonical_sha256; print(canonical_sha256(json.load(open(sys.argv[1]))))' configs/experiments/qwen_agent_search.json)

$PYTHON_BIN scripts/collect_qwen_sft_inputs.py \
  --config configs/experiments/qwen_agent_search.json \
  --trajectory-run-plan results/eval/qwen_agent_search/run_plans/trajectory_q9.json \
  --output results/eval/qwen_agent_search/trajectories/base_inputs.json

$PYTHON_BIN scripts/build_qwen_agent_sft.py \
  --phase counterfactuals \
  --train-manifest results/eval/qwen_agent_search/frozen/train600.jsonl \
  --input-bundle results/eval/qwen_agent_search/trajectories/base_inputs.json \
  --config-sha256 "$CONFIG_SHA" \
  --output-dir results/eval/qwen_agent_search/trajectories/counterfactual_specs
```

按三个数据集分别通过 `launch_qwen_counterfactuals.sh` 启动 `run_qwen_counterfactuals.py` 参数；启动器强制 `setsid + nohup + --resume`、单请求 80 秒和 90 秒结果停滞保护。完成后再次调用 `collect_qwen_sft_inputs.py`，为每个反事实 JSONL 增加一个 `--counterfactual` 参数。然后用新 bundle 运行 `build_qwen_agent_sft.py --phase select`。bundle 会校验实验配置、run plan、所有轨迹文件、Train600、三个 Train200、9B artifact、Agent config 和 runner fingerprint；任一文件或哈希变化都会拒绝构建或续跑。`--phase select` 的训练文件始终是其 `--output-dir` 下实际生成的 `sft.jsonl`，例如：

```bash
$PYTHON_BIN scripts/build_qwen_agent_sft.py \
  --phase select \
  --train-manifest results/eval/qwen_agent_search/frozen/train600.jsonl \
  --input-bundle results/eval/qwen_agent_search/trajectories/all_inputs.json \
  --config-sha256 "$CONFIG_SHA" \
  --output-dir results/eval/qwen_agent_search/trajectories/selected
```

训练前 GPU 4–7 必须全部空闲：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
SFT_ENV_DIR=$PROJECT_DIR/.venv-swift \
  bash scripts/train_qwen_agent_9b_lora.sh \
  --smoke \
  --train-data results/eval/qwen_agent_search/trajectories/selected/sft.jsonl \
  --output-dir results/eval/qwen_agent_search/sft_checkpoints/qwen35_9b_lora_smoke

CUDA_VISIBLE_DEVICES=4,5,6,7 \
SFT_ENV_DIR=$PROJECT_DIR/.venv-swift \
  setsid nohup bash scripts/train_qwen_agent_9b_lora.sh \
  --train-data results/eval/qwen_agent_search/trajectories/selected/sft.jsonl \
  > logs/qwen_agent_sft.log 2>&1 < /dev/null &
```

训练脚本在占用 GPU 前验证模型 artifact、`ms-swift==4.4.2`、真实 Qwen3.5 模板 loss mask；只有 plan/tool/memory/stop/final assistant token 计算 loss。正式三轮训练前必须先执行 `--smoke`，它强制使用显式、独立于正式 checkpoint 的输出目录并只反向传播 1 step；smoke 成功后再启动上面的正式任务。

每个 epoch 用 `freeze_qwen_sft_checkpoint.py` 冻结 LoRA，然后用 `LORA_PATH`、`LORA_SERVED_NAME` 启动服务并运行：

```bash
PYTHON_BIN=$PYTHON_BIN bash scripts/launch_qwen_agent_phase.sh \
  configs/experiments/qwen_agent_search.json sft_dev q9 \
  --frozen-winner-config results/eval/qwen_agent_search/frozen/q9_agent_winner.json \
  --sft-checkpoint-config /path/to/frozen_epoch_checkpoint.json
```

三个 epoch 都完成后用 `select_qwen_sft_checkpoint.py` 比较共同 Dev150。只有正确数严格更高、平均总 Token 与视觉 Token 均不超过 Teacher 的 70%、失败率不超过 1% 且泄漏为 0，才会生成 `sft_winner.json`。

## 5. 最终固定 300 条

冻结未训练 winner 后即可分别运行 q9、q4 最终矩阵；即使 SFT 没有通过门槛，
这两组也必须完成并真实报告。只有 sft9 组要求 gate-passed `sft_winner.json`。
三组必须分别启动正确服务和任务：

```bash
PYTHON_BIN=$PYTHON_BIN bash scripts/launch_qwen_agent_phase.sh \
  configs/experiments/qwen_agent_search.json final_matrix q9 \
  --frozen-winner-config results/eval/qwen_agent_search/frozen/q9_agent_winner.json
```

将 `q9` 替换为 `q4` 即可运行 4B 组。运行 `sft9` 时再额外传入
`--frozen-sft-winner results/eval/qwen_agent_search/frozen/sft_winner.json`，并在每组之间停止旧服务。三组完成后，将已有的 `final_matrix_*.json` run plan 传给 `summarize_qwen_final_matrix.py`。

最终矩阵固定包含：4B/9B 真无视频、4B/9B 最佳 Direct、4B/9B EVA-clean、同配置 4B/9B 最佳未训练 Agent、最终 SFT-9B。报告输出 raw/accessible/common-valid accuracy、McNemar、改对/改错、完整 Token、延迟和失败分类；未达标时按真实数值报告。
