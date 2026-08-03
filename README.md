# FlashVID for Qwen3.5-4B + vLLM

This repository reproduces the **vision-encoder / before-LLM** compression
path from [FlashVID](https://arxiv.org/abs/2602.08024) on
[`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B), packaged as a
[vLLM out-of-tree model plugin](https://docs.vllm.ai/en/v0.25.1/contributing/model/registration/).

Only one FlashVID compression method is implemented:

- DySeg: dynamic temporal segmentation
- ADTS: attention-and-diversity token selection
- TSTM: tree-based spatiotemporal token merging
- DPC-kNN: spatial contextual aggregation

There is **no LLM-layer token pruning**. The codebase intentionally contains
no `fastv_prune`, `pruning_layer`, or `llm_retention_ratio`.

## Interface

```bash
flashvid-serve /path/to/Qwen3.5-4B \
  --vision-retention-ratio 0.10 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 32768 \
  --port 8000
```

`--vision-retention-ratio 0.10` means retaining approximately 10% of the
vision-encoder tokens. The value is fixed for the lifetime of the service and
must be in `(0, 1]`. A ratio of `1.0` bypasses compression exactly.

The service remains OpenAI-compatible:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/Qwen3.5-4B",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "video_url", "video_url": {"url": "https://example/video.mp4"}},
        {"type": "text", "text": "Describe the video."}
      ]
    }],
    "temperature": 0,
    "max_tokens": 64
  }'
```

## How the vLLM integration works

The plugin registers `FlashVIDQwen3_5ForConditionalGeneration` without
patching the installed vLLM package. It captures Q/K from the last
Qwen3.5 vision-attention layer, computes per-frame received attention in
bounded query chunks, and applies ADTS + TSTM after the vision merger.

The retained/merged token's anchor index is then used to update:

- video placeholder count and ordering;
- M-RoPE temporal/spatial positions;
- all concatenated visual channels (including DeepStack channels if present).

Qwen3.5-4B's current official configuration has an empty
`deepstack_visual_indexes`, but the compressor operates on the full concatenated
visual feature width.

For compatibility with vLLM's dynamic multimodal placeholder path, a video
retains at least one frame's token count. Images and pure-text requests are not
compressed. This is a video vision-encoder optimization.

## Local tests

The compression core only needs PyTorch:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

The tests verify released-implementation ADTS behavior, deterministic DPC-kNN,
TSTM merging, exact requested budgets, and the ratio-1 bypass. They also scan
the source tree to prevent accidental introduction of inner-LLM pruning.

## Server installation

All commands below keep the environment, model, caches, logs, and results
inside the cloned project directory.

```bash
cd /data02/usr/wangqihao/Demo/test
git clone https://github.com/Seven-creater/flashvid.git flashvid
cd flashvid
bash scripts/setup_server.sh
bash scripts/download_model.sh
```

`setup_server.sh` installs the CUDA 13 forward-compatibility userspace library
inside `.venv` when required by the pinned vLLM wheel. It does not replace the
host driver or require root. The launcher also binds the wheel-provided CUDA
toolkit for JIT compilation and redirects vLLM, FlashInfer, and Torch caches to
the project `.cache/` directory.

Start the recommended 8×A6000 throughput configuration:

```bash
bash scripts/serve_dp8.sh 0.10
```

Run a smoke request and a concurrent benchmark:

```bash
.venv/bin/python scripts/smoke_openai.py \
  --video-url https://example/video.mp4

.venv/bin/python scripts/benchmark_openai.py \
  --video-url https://example/video.mp4 \
  --concurrency 32 --requests 128 \
  --output results/benchmark.json
```

Use `scripts/sweep_server.py` to compare DP and scheduler settings. It launches
one configuration at a time, rejects failed/OOM runs, and writes the
highest-throughput result to `results/best_config.json`.

## Fixed reproduction parameters

| Parameter | Value |
|---|---:|
| ADTS fraction (`alpha`) | 0.7 |
| TSTM temporal threshold | 0.8 |
| DySeg transition threshold | 0.9 |
| Minimum segments | 4 |
| Token selection | `attn_div` |
| Inner-LLM expansion | not used |

## Version scope

- `vllm==0.25.1`
- `Qwen/Qwen3.5-4B`
- Main deployment: DP=8, TP=1

The plugin deliberately targets the vLLM 0.25.1 Qwen3.5 implementation. Pinning
is intentional because the integration uses model-internal multimodal hooks.

## Unified long-video evaluation

The evaluator normalizes CG-Bench, LVBench, and LSDBench, freezes deterministic
manifests, and writes resumable JSONL results. Direct, EVA-style Agent,
frozen-candidate Hybrid v3c, and FlashVID Hybrid share one entry point. Model
requests never receive benchmark answers, clue intervals, annotated time
ranges, or human question types.

```bash
python -m pip install -e ".[test]"
python scripts/evaluate_mcq.py \
  --dataset lsdbench \
  --backend direct \
  --annotations /path/to/LSDBench/test.json \
  --video-root /data02/pretrained_model/cvr_learn/cvr_data/07_lsdbench/videos \
  --model Qwen3.5-9B \
  --base-url http://127.0.0.1:8001/v1 \
  --concurrency 32 \
  --sample 100 --seed 42 --resume \
  --output-dir results/eval
```

The official EVA frame selector is pinned as a Git submodule. The dependency
lock records that the archived local snapshot `1774939` and fetchable upstream
commit `758ad8d` have the identical Git tree. Hybrid v3c can
reuse a frozen Direct candidate without rerunning Direct:

```bash
python scripts/evaluate_mcq.py \
  --dataset lsdbench \
  --backend hybrid_frozen \
  --agent-version hybrid_v3c \
  --candidate-results results/eval/lsdbench_direct.jsonl \
  --manifest results/eval/lsdbench_manifest_42_100.jsonl \
  --annotations /path/to/LSDBench/test.json \
  --video-root /path/to/LSDBench/videos \
  --resume --output-dir results/eval/hybrid_v3c
```

## Training-free budget sweep

`flashvid_hybrid` keeps the 9B controller text-only and sends selected clips to
the FlashVID-4B perception bank. The sweep compares fixed 10/25/50/100%, seeded
random, question-route rules, evidence-driven escalation, and a frozen
cost-matched random baseline. It also freezes three prompt IDs: `legacy_v1`,
`budget_rubric_v1`, and `budget_escalation_v1`.

```bash
python scripts/run_budget_sweep.py \
  --config configs/experiments/budget_sweep_dev.json \
  --stage smoke --controller q9 --resume
```

The launcher validates file and configuration hashes, creates the 10-sample
smoke manifests, and uses `setsid + nohup` on Linux. It never starts training or
model services. Runs containing more than one controller must be launched in
separate `--controller q9`, `--controller q4base`, and `--controller ck39`
phases so a stopped or mismatched service cannot silently corrupt a sweep.
Summaries include candidate-only at zero incremental FlashVID
visual cost and compute separate system, 9B-controller, and 4B-controller
Pareto frontiers:

```bash
python scripts/summarize_budget_sweep.py \
  --config configs/experiments/budget_sweep_report.json
```

The frozen execution order is base smoke/dev, base summary, three
cost-matched-random smoke/dev runs, combined summary, and finally the original
300-sample test. The matched and final templates intentionally contain SHA
placeholders and refuse to run until the preceding Dev summaries are frozen;
the test runner selects executable policies only from the separate 9B and 4B
Dev frontiers, never from the system-level Direct/v3c reference frontier.

Service helpers are limited to the FlashVID budget bank and Qwen3.5-9B
launchers. Large models, datasets, caches, raw results, and server environment
files are intentionally ignored by Git. The default bank layout remains GPUs
0–3 for perception and 4–7 for the DP=4 controller. On a shared server it can
be remapped without editing code through `BUDGET_GPU_R010`,
`BUDGET_GPU_R025`, `BUDGET_GPU_R050`, `BUDGET_GPU_R100`,
`BUDGET_GPU_CONTROLLER`, and `CONTROLLER_DATA_PARALLEL_SIZE`; ownership checks
still refuse to overwrite unrelated PIDs or occupied ports. If two perception
services intentionally share one GPU, set `PERCEPTION_KV_CACHE_MEMORY_BYTES`
to an explicit per-service allocation because vLLM's percentage-based cache
autotuning assumes exclusive ownership of the device.

## Attribution

The compression algorithm is adapted from
[Fanziyang-v/FlashVID](https://github.com/Fanziyang-v/FlashVID), released under
Apache-2.0. The vLLM integration reuses public extension points and the
multimodal pruning/M-RoPE protocol from vLLM 0.25.1.
