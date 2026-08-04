# Fast Hybrid EVA bulk launchers

The Teacher and Judge launchers are deliberately separate from the evaluator.
They only validate a frozen plan, assign whole jobs to the configured 8200/8201
services, and invoke the existing evaluator/Judge scripts. They never start,
stop, or inspect model services.

## Teacher matrix

Use the complete `base_plan.jsonl` for the base phase, including on resume. The
launcher rejects a partial base plan because a schedule is evaluated against
the immutable Train200 manifest. Rescue plans may contain subsets; the
launcher freezes schedule-specific manifest and candidate subsets while
recording their hashes.

```bash
cd /data02/usr/wangqihao/Demo/test/qwen_agent_search
export PYTHONPATH="$PWD/src"
PYTHON=/data02/usr/wangqihao/Demo/test/flashvid/.venv/bin/python
CONFIG=configs/experiments/fast_hybrid_eva_sft.json
CONFIG_SHA=$(sha256sum "$CONFIG" | awk '{print $1}')

$PYTHON scripts/launch_fast_hybrid_teacher_matrix.py \
  --config "$CONFIG" \
  --expected-config-sha256 "$CONFIG_SHA" \
  --specs results/eval/fast_hybrid_eva_sft/trajectories/base_plan.jsonl \
  --candidate-results lvbench=/absolute/path/lvbench_train_direct.jsonl \
  --candidate-results lsdbench=/absolute/path/lsdbench_train_direct.jsonl \
  --candidate-results cgbench=/absolute/path/cgbench_train_direct.jsonl \
  --python "$PYTHON" \
  --repo-root "$PWD" \
  --concurrency-per-endpoint 16 \
  --timeout 80 \
  --resume \
  --print-nohup-command
```

The final flag prints a directly executable `setsid nohup ... &` command. Run
that printed command once. Without `--print-nohup-command`, the launcher runs
one evaluator child at a time on each endpoint; the two endpoint queues run in
parallel, giving a maximum inference concurrency of 32.

Each `(schedule, dataset)` has its own result, frame, and log directory. The
launcher freezes a command plan before inference and audits exact sample IDs,
deferred scoring, generation seed, schedule, manifest hash, and
`candidate_rerun=0` after all children finish. A missing candidate file or
candidate sample stops the matrix before any model request.

Teacher model/parse failures are immutable outcomes and are never retried by
the bulk launcher. Each API request already receives the one frozen, same-seed
immediate retry inside the Fast Hybrid adapter; `--resume` only fills samples
that have no committed row.

## Candidate-blind Judge matrix

First run `prepare_fast_hybrid_judges.py`. Its filtered specs and trajectories
are the only Judge inputs. Empty schedules are not launched.

```bash
$PYTHON scripts/launch_fast_hybrid_judge_matrix.py \
  --config "$CONFIG" \
  --expected-config-sha256 "$CONFIG_SHA" \
  --specs results/eval/fast_hybrid_eva_sft/trajectories/prejudge/judge_specs.jsonl \
  --trajectories results/eval/fast_hybrid_eva_sft/trajectories/prejudge/judge_trajectories.jsonl \
  --python "$PYTHON" \
  --repo-root "$PWD" \
  --concurrency-per-endpoint 16 \
  --timeout 80 \
  --resume --retry-failed-processes \
  --print-nohup-command
```

The Judge launcher assigns each non-empty schedule to one endpoint, keeps at
most one child active per endpoint, and passes the frozen Judge seeds and
temperature from the experiment config. Its final audit requires exactly one
row per eligible trajectory and all three Judge seeds, while retaining
`complete_with_failures` as an explicit model/infrastructure outcome for the
offline selector.

Both launchers support `--dry-run`. Dry-run validates all existing inputs and
prints the exact child commands without writing plans, subsets, or results.
The configuration digest is the SHA-256 of the file bytes (the value printed by
`sha256sum`); it must be the same digest used when the trajectory specs were
created.
