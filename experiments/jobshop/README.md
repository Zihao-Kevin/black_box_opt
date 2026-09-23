# JobShop: 70 runs, 128-instance validation

This directory publishes the Kings campaign `kings-beta010-val128-n10`: seven estimators, ten new training seeds (10–19), 32 epochs, and dispatch entropy coefficient 0.1. It does **not** pool the earlier 9-seed/64-validation campaign or the later delta-only hyperparameter sweep.

See [RESULTS.md](RESULTS.md) for the verified final table and [results/](results/) for machine-readable evidence. The learned-Q variants are **online QCV** and **online Q + row delta**, both initialized from the same offline reward dataset.

## Task and protocol

The input contains one full JobShop instance. A frozen Qwen2.5-Coder-7B-Instruct backbone with trainable rank-8 LoRA on the final MLP down projection emits a constrained canonical JSON plan. Its 1,500 legal actions combine 15 dispatch rules, 10 tie-breakers, and 10 valid postprocessing/radius choices. A deterministic constructor produces a schedule; the pinned Frontier validator checks feasibility. Reward is `max(max_job_work, max_machine_work) / makespan`.

The frozen dataset has 128 train, 128 validation and 128 test instances, balanced across eight sizes and four families. The validation set extends the original 64 instances without altering original train/test instances. All methods share the same seed-specific actor initialization, instance ordering and sampling seed; policies need not sample the same actions. The initial Q fit is shared across seeds.

Each update uses two instances and eight samples per instance. There are 64 updates per epoch, 2,048 updates and 32,768 training evaluations per run. Initial and all 32 epoch actors are evaluated on validation with four samples per instance. A final mismatched-input diagnostic adds another validation pass: 17,408 validation/diagnostic calls per run. Epoch 0–32 selection maximizes validation mean reward; ties select the earlier epoch. The selected checkpoint is frozen before test (128 instances × four samples = 512 calls/run). Test sampling RNG is `10000 + 1000 * instance_index`, identical across runs.

Adam learning rate is 1e-5. All methods receive the exact dispatch-marginal entropy gradient with coefficient 0.1. Q starts from 8,023 unique offline labels, immediately records committed on-policy observations, and refits for 1,000 steps after epochs 1–31. No validation or test label enters Q. Q initialization artifacts are included under `artifacts/offline-q/`.

**Comparison limits:** GRPO retains sequence-length normalization, group-standardized rewards and reference KL coefficient 0.01, while the other methods do not share those extra transformations. Thus these are matched sampling budgets, not identical objective scales. Q-based methods have an offline-label cost that baselines may not use. Logged gradient norms are **not** gradient variances. Standard deviations are across training seeds on the fixed evaluation set, not uncertainty over new datasets. The test set was also used in earlier development comparisons; this is not a newly acquired independent benchmark.

## Installation and data

Use Linux, Python 3.12, CUDA-capable PyTorch and Docker. The original run used H100 GPUs. Run commands **from this directory**. The dependency freeze and historical source snapshots are retained with the results.

```bash
python3.12 -m venv .venv-train
.venv-train/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements-train.lock
bash scripts/provision_validator.sh
.venv-train/bin/pip install wandb==0.30.0  # optional monitoring
```

Obtain `Qwen/Qwen2.5-Coder-7B-Instruct` revision `c03e6d358207e414f1eca0bb1891e29f1db0e242` using your Hugging Face setup before preparing features; the feature builder intentionally uses local-files-only model loading. Model weights, feature tensors, credentials and large training checkpoints are not committed.

Restore the included Q artifacts into the historical paths expected by the trainer:

```bash
mkdir -p runs/offline-q-v1/data runs/offline-q-v1/model
cp artifacts/offline-q/pool.json runs/offline-q-v1/data/
cp artifacts/offline-q/{selected-tables,complete}.json runs/offline-q-v1/model/
```

Prepare the 256 train/validation feature files on an allocated GPU (approximately 18 GB cache; no test features are read):

```bash
CUDA_VISIBLE_DEVICES=YOUR_GPU_UUID .venv-train/bin/python -m jobshop_rl.row_prepare_sharded \
  --config configs/synthetic-val128-v1.json --out /your/local/cache/jobshop-val128
```

## Training and resuming

The portable launcher takes an explicit list of GPUs already allocated to you. It creates 70 workers (ten per method), limits concurrent Docker evaluations, and preserves per-run resume state. It does not request scheduler allocations. Do not run it against GPUs that are not yours.

```bash
.venv-train/bin/python scripts/launch_campaign.py \
  --gpus GPU_UUID_1,GPU_UUID_2,GPU_UUID_3,GPU_UUID_4,GPU_UUID_5,GPU_UUID_6,GPU_UUID_7 \
  --cache /your/local/cache/jobshop-val128 --out runs/reproduction-val128
```

For a smoke run or one method:

```bash
CUDA_VISIBLE_DEVICES=YOUR_GPU_UUID JOBSHOP_EVALUATOR=persistent-v1 \
.venv-train/bin/python -m jobshop_rl.kings_val128_train \
  --config configs/synthetic-val128-v1.json --cache /your/local/cache/jobshop-val128 \
  --out runs/smoke-row --method row_delta --seed 10 \
  --slot-dir /your/local/solver-slots --docker-slots 1 --stop-after 2
```

Rerun the same command without `--stop-after` to continue. Checkpoint replacement is the commit boundary: optimizer, actor, RNG, Q and baseline states resume together, with interrupted journal writes reconciled. The initial and every epoch actor are retained, plus latest full state and validation-best actor. Partial validation resumes at committed instance groups. Failures stop the run instead of becoming low rewards.

The persistent evaluator keeps one restricted container per worker but uses a fresh Python process and clean temporary directory for each candidate. Network is disabled; memory, CPU and process limits apply. This backend is for the trusted canonical plan constructor, not a general arbitrary-code sandbox.

## Test and monitoring

After all 70 runs finish, evaluate the validation-selected actors without any test-based reselection:

```bash
CUDA_VISIBLE_DEVICES=YOUR_GPU_UUID .venv-train/bin/python scripts/evaluate_campaign.py \
  --run-root runs/reproduction-val128 --cache /your/local/cache/jobshop-val128 \
  --stream-cache /your/local/cache/jobshop-test-stream
```

Test backbone features are streamed and removed per instance. The evaluator uses 32 CPU solver processes. The standalone wrapper is intended for seeds 10–19 and all seven methods, matching this campaign.

Optional W&B monitoring uses existing login credentials; never put credentials in this repository:

```bash
JOBSHOP_RUN_ROOT=runs/reproduction-val128 .venv-train/bin/python scripts/wandb_monitor.py \
  --seed 10 --beta 0.10 --method row_delta --entity YOUR_ENTITY --project YOUR_PROJECT
```

The original group was `kings-beta010-val128-n10`. Monitoring reads files and does not change training. Historical launchers in `scripts/historical/` preserve the actual Kings paths and hardware orchestration; they are provenance, **not portable entry points**.

## Validation and reproduction of reported results

```bash
PYTHONPATH=.:tests OMP_NUM_THREADS=1 .venv-train/bin/python -m pytest -q tests
python scripts/summarize_results.py
```

The summary script recomputes validation-selected test scores from the committed per-instance test records and verifies them against the exported summaries. All result figures concern the fixed validation set. Training curves report epoch-average sampled training rewards; these should not be compared directly to held-out reward as if they used identical sampling.
