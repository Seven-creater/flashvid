# Role-separated Process SFT runbook

This runbook freezes the experiment-control layer only. It does not change the
Perception-Memory EVA runtime, its SFT exporter, or result reporting.

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

- Controller: the only trainable LoRA; text-only, with zero media inputs.
- Perception: frozen Qwen3.5-9B base; current frame batch only; candidate blind.
- Judge family: frozen Qwen3.5-9B base for completeness, evidence answer, and
  confirmation; candidate blind.
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
  --output-dir results/eval/perception_memory_eva_sft/dev_role_ablation
```

This writes the frozen 2^3 Controller/Perception/Judge matrix at seed 42. The
files are orchestration skeletons marked `requires_role_endpoint_router`; do
not execute them until the existing runtime can bind role-specific endpoints
without changing its prompt or tool semantics. The materializer rejects every
split other than Dev. It never reads, copies, or scores Test300.

The role matrix diagnoses where the old shared LoRA causes regressions. It may
select the role boundary for training, but it may not select or edit prompts.

## 4. Train and evaluate on Dev

Use the existing ms-swift trainer and Perception-Memory exporter after the role
binding has been supplied. Preserve the five frozen process families:

1. direct answer;
2. single frame selection;
3. timestamp-grounded selection;
4. hierarchical refinement;
5. multi-interval exploration.

Incomplete prefixes target `continue + next frame_select`; only evidence-only
3/3 correct complete prefixes may target stop/final answer. Tool observations,
images, user input, and hidden reasoning remain masked.

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
