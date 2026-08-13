# Role-separated Process SFT runbook

This runbook covers the frozen four-role runtime, visual-only completeness
labels, complete Process-SFT episodes, and experiment controls.

## 1. Validate and lock primary sources

Run these commands from the repository root:

```bash
export PYTHONPATH=src:.
python scripts/validate_role_separated_process_sft.py \
  --config configs/experiments/role_separated_process_sft.json
python scripts/generate_role_separated_source_lock.py \
  --config configs/experiments/role_separated_process_sft.json \
  --output configs/source_locks/role_separated_process_sft_sources.lock.json
```

The lock contains exactly seven official repositories, full commit IDs, and an
explicit reuse/exclusion boundary for every source. Repository code must be
checked out on the server at the locked commit. Models, data, and dependencies
must not be downloaded to the local PC; server downloads use hf-mirror or a
domestic mirror.

## 2. Keep roles separated

- Planner: the only trainable LoRA; text-only, with zero media inputs.
- Observer: frozen Qwen3.5-9B base; current frame batch only; candidate blind.
- Verifier: frozen Qwen3.5-9B base; public question/options plus accumulated
  real frames only; no ledger, candidate, annotation, or prior reasoning.
- Answerer: frozen Qwen3.5-9B base; ledger plus at most 16 Verifier-cited
  decisive frames; candidate blind.
- A single adapter must never be mounted on all roles in the selected training
  run. The legacy shared-adapter cells exist only for Dev attribution.

Training quantity is an audit field, not a stop condition. Record available
questions, prefixes, and candidate-fix questions in `training_quantity`; null
or low values are reported but never change gate pass/fail. Data quality,
leakage, runtime identity, accuracy, Token, and engineering failures remain
blocking.

## 3. Materialize the Dev-only role ablation

```bash
python scripts/materialize_role_ablation_matrix.py \
  --config configs/experiments/role_separated_process_sft.json \
  --split dev \
  --derive-dev10 \
  --output-dir results/eval/role_separated_process_sft/role_ablation
```

This verifies every frozen Dev50 SHA, derives its first ten rows atomically,
and writes the six pre-registered Dev30 cells at seed 42: all Base, each old
LoRA role in isolation, and all four roles on the old LoRA. Each run directory
contains an exact `role_config.json` accepted by `--pm-role-config`. First run
`base_all`, then freeze its label-free role inputs:

```bash
python scripts/freeze_role_ablation_inputs.py \
  --input BASE_DEV30_RESULTS.jsonl \
  --output results/eval/role_separated_process_sft/role_ablation/frozen_role_inputs.jsonl
```

Planner-only and all-role cells are end-to-end system interventions. Observer,
Verifier, and Answerer cells use `scripts/run_role_ablation_pair.py`: both arms
receive the same request SHA, frame paths, timestamps, and per-frame content
SHA. Observer pairs can additionally pass frozen Base Verifier/Answerer
bindings to report the conditional downstream answer. These fixed-schedule
results are conditional role effects, not full-runtime causal effects. The
materializer rejects every split other than Dev and never reads or scores
Test300.

Each fixed-role command must be bound to its materialized run file and SHA; the
endpoint, model, and artifact identities are read from that immutable file, not
accepted as free command-line overrides:

```bash
RUN_JSON=results/eval/role_separated_process_sft/role_ablation/observer_old_lora/run.json
RUN_SHA=$(sha256sum "$RUN_JSON" | awk '{print $1}')
FROZEN=results/eval/role_separated_process_sft/role_ablation/frozen_role_inputs.jsonl
FROZEN_SHA=$(sha256sum "$FROZEN" | awk '{print $1}')
python scripts/run_role_ablation_pair.py \
  --input "$FROZEN" \
  --expected-input-sha256 "$FROZEN_SHA" \
  --output results/eval/role_separated_process_sft/role_ablation/observer_old_lora/paired.jsonl \
  --run-config "$RUN_JSON" \
  --expected-run-sha256 "$RUN_SHA" \
  --local-media-paths
```

The role matrix diagnoses where the old shared LoRA causes regressions. It may
select the role boundary for training, but it may not select or edit prompts.

## 4. Train and evaluate on Dev

Use the existing ms-swift trainer and Perception-Memory exporter after the role
binding has been supplied. Preserve the four frozen visual path families:

1. single frame selection;
2. timestamp-grounded selection;
3. hierarchical refinement;
4. multi-interval exploration.

Every training question must execute at least one real `frame_select`; a
candidate-only/direct-answer trajectory is not a Process-SFT training sample.

Run `scripts/judge_perception_memory_visual_csv.py`, then
`scripts/select_perception_memory_visual_csv_trajectories.py`; the latter joins
labels offline and deterministically searches the real stable-trajectory pool
for a subset satisfying all three frozen quality contracts. It never copies or
synthesizes a STOP. The build must stop when no real subset can satisfy:

- frozen-candidate correct/wrong strata at 1:1 (within 10%);
- observed incomplete CONTINUE and STOP decisions at 1:1 (within 10%);
- all four visual-path families non-empty with max/min no greater than 1.1.

Visual-CSV provenance includes ordered paths, timestamps, and the SHA-256 of
every frame file. Any byte change between judging, offline labeling, and build
is a hard failure. Incomplete prefixes target `continue + next frame_select`;
only visual-only 3/3 correct complete prefixes may target STOP. Tool
observations, images, user input, and hidden reasoning remain masked.

Use `scripts/gate_perception_memory.py` for the existing strict gate. The Dev
payload may include non-blocking counts:

```json
{
  "phase": "dev",
  "training_quantity": {
    "observed_questions": 234,
    "observed_prefixes": 468,
    "candidate_fix_questions": 40
  }
}
```

The rest of the payload is the existing baseline/candidate path schema. Dev
passes only when the candidate uses the same runtime as the untrained baseline,
gains at least 3/150 on the three-seed mean, does not regress any dataset, uses
at most 70% of both total and visual Tokens, and has at most 1% engineering
failures. Annotation leak, candidate rerun, and duplicate sample IDs remain
zero.

Long stages use `setsid nohup ... --resume`; do not introduce a recurring
monitor.

## 5. Test is a one-way gate

Do not expose Test300 to role ablation, prompt selection, checkpoint selection,
or training. Freeze exactly one Dev winner and its passed Dev-gate JSON plus
SHA-256. Only then may the existing Test gate run once.

Test passes only if the frozen winner:

- gains at least 6/300 over the same-runtime untrained baseline;
- reaches at least 157/300;
- does not regress any dataset and reaches LVBench 45, LSDBench 63, CG-Bench 43;
- uses at most 70% of both total and visual Tokens;
- has at most 1% engineering failures and zero leak/rerun/duplicate-ID events.

Failure on Dev ends the experiment. Test results must never be used to change a
prompt, role assignment, checkpoint, or training corpus.
