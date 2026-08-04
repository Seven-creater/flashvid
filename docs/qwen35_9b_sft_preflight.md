# Qwen3.5-9B LoRA 训练预检

这套入口只负责训练环境和一次性 smoke，不会修改 Fast Hybrid、轨迹筛选或冻结结果。

## 1. 创建独立环境

服务器必须提供 `python3.12`。安装器创建独立的
`.venv-qwen35-sft-cu121`，不继承推理环境，也不覆盖已有 `.venv`：

```bash
bash scripts/install_ms_swift_442.sh
```

安装器先从 PyTorch 官方 CUDA 12.1 索引安装
`torch==2.5.1`，再安装
`configs/training/qwen35_9b_sft_cuda121.lock.txt` 中的固定依赖，并执行
`pip check`。安装报告写入 `.runtime/ms_swift_442/installed.json`。

## 2. 完整预检

完整预检会按顺序执行：

1. 只尝试停止带本仓库 ownership marker 的 8200/8201 服务；
2. 自动选择空闲的 GPU 0–7，若不能独占则退到 GPU 4–7；
3. 验证每张卡总显存至少 42 GiB、CUDA 12.1、BF16 和固定依赖；
4. 通过 Transformers 加载本地 `Qwen3_5ForConditionalGeneration` 权重；
5. 用训练 JSONL 的第一条样本执行一次、且仅一次反向传播。

```bash
bash scripts/preflight_qwen35_9b_lora.sh \
  --train-data results/eval/qwen_agent_search/trajectories/selected/sft.jsonl \
  --output-dir results/eval/qwen_agent_search/sft_checkpoints/qwen35_9b_lora_smoke
```

ownership、UID、vLLM 命令或端口任一项不匹配时，停止操作会直接失败；脚本不会使用
`pkill`、`kill -9`，也不会停止其他 PID。若 GPU 上仍有其他进程，预检退出并保留这些进程。

## 3. 正式训练

预检通过后才能运行正式任务。默认不停止任何服务；调用者应先显式安排 GPU：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
setsid nohup bash scripts/train_qwen_agent_9b_lora.sh \
  --train-data results/eval/qwen_agent_search/trajectories/selected/sft.jsonl \
  > logs/qwen_agent_sft.log 2>&1 < /dev/null &
```

仅支持两种布局：8 卡时梯度累积 4，4 卡时梯度累积 8；每卡 batch 为 1，因此有效
batch 均为 32。训练固定 16K 上下文、LoRA rank 16/alpha 32/dropout 0.05，冻结
ViT 与 Aligner，并在启动前用真实 Qwen3.5 模板验证逐消息 loss mask。
