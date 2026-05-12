# `scripts/bench/` — Small-data performance comparison

A minimal harness for comparing the **co-located baseline**
(`run_logit_fusion_gspo.sh`) vs. the **asymmetric remote-teacher** layout
(`run_logit_fusion_gspo_remote.sh`) on the same machine.

Each invocation produces an isolated run directory containing:

| File | What it is |
|---|---|
| `config.env`         | Snapshot of the env vars + flags that drove the run |
| `walltime.csv`       | Phase markers (script_start, trainer_launch, trainer_exit, …) |
| `gpu_metrics.csv`    | `nvidia-smi` poll, one row per GPU per sample |
| `trainer.log`        | Full stdout/stderr of the training wrapper |
| `teacher_server.log` | Remote mode only — copied from the teacher's stdout |
| `training_output/`   | Whatever the trainer writes (HF Trainer's output_dir) |

A separate `summarize.py` prints a side-by-side table of the parsed metrics.

---

## 1. Quick start

```bash
conda activate trl
cd logit-fusion

# 1. baseline (co-located teacher + student on the visible GPUs)
bash scripts/bench/run_bench.sh baseline               # PROFILE=tiny

# 2. remote (asymmetric: student on STUDENT_GPUS, teacher on TEACHER_GPUS)
bash scripts/bench/run_bench.sh remote                 # PROFILE=tiny

# 3. compare
python scripts/bench/summarize.py outputs/bench/baseline__* outputs/bench/remote__*
```

`outputs/bench/<MODE>__<PROFILE>__<timestamp>[__$RUN_TAG]/` is created per run.
The most recent runs are matched by the `*` glob above; you can also pass exact
paths.

---

## 2. Load profiles (`tiny` / `stress` / `xl`)

The second positional argument selects how heavy the workload is. The `tiny`
default keeps each run under ~90 s for fast iteration; `stress` and `xl` run
longer so steady-state behavior dominates the noise of first-step warmup and
amplifies the per-token RPC trade-off in remote mode.

| Profile | `MAX_STEPS` | `PER_DEVICE_TRAIN_BATCH_SIZE` | `NUM_GENERATIONS` | `MAX_PROMPT_LENGTH` | `MAX_COMPLETION_LENGTH` | Notes |
|---|---|---|---|---|---|---|
| `tiny`   |  4 | 2 | 2 |  512 |  256 | smoke test |
| `stress` |  8 | 4 | 4 |  512 | 1024 | longer completions — where the asymmetric layout's win shows up |
| `xl`     | 12 | 4 | 8 | 1024 | 2048 | heaviest preset that still fits 0.5B+7B on 4×80G |

```bash
# Stress test — longer completions amplify the per-token parallelism win
bash scripts/bench/run_bench.sh baseline stress
bash scripts/bench/run_bench.sh remote   stress

# Heaviest preset
bash scripts/bench/run_bench.sh remote   xl

# Compare a specific profile
python scripts/bench/summarize.py outputs/bench/baseline__stress__* outputs/bench/remote__stress__*
```

These knobs are still defaults: any env var you set on the caller side
(`MAX_STEPS=20 …`) overrides whatever the profile picked. You can also
combine them — e.g. "stress profile but with batch 8":

```bash
PER_DEVICE_TRAIN_BATCH_SIZE=8 bash scripts/bench/run_bench.sh remote stress
```

The remaining knobs that no profile touches:

| Knob | Default |
|---|---|
| `STEPS_PER_GENERATION`        | `2` |
| `GRADIENT_ACCUMULATION_STEPS` | `1` |
| `ALPHA_INIT`                  | `0.5` |
| `eval_strategy`               | `no` (forced off by the harness) |
| `save_strategy`               | `no` (forced off by the harness) |
| `logging_steps`               | `1` |

The model defaults are unchanged from the real run scripts:
`Qwen/Qwen2.5-0.5B` (student) + `Qwen/Qwen2.5-7B` (teacher). Override with
`MODEL_NAME_OR_PATH=...` and `TEACHER_MODEL_NAME_OR_PATH=...` if you want a
faster sanity test (e.g., `TEACHER_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-1.5B`).

---

## 3. GPU partitioning — choosing what "fair" means

The **remote** mode always partitions hardware explicitly via `STUDENT_GPUS`
and `TEACHER_GPUS`. Defaults (4-GPU box):

```
STUDENT_GPUS=0
TEACHER_GPUS=1,2,3
```

The **baseline** has three strategies, selected via `BASELINE_STRATEGY`. Each
sets `NUM_PROCESSES` and `BASELINE_GPUS` (which controls
`CUDA_VISIBLE_DEVICES` for the trainer) for you.

| Strategy | `NUM_PROCESSES` | `BASELINE_GPUS` | What you're measuring |
|---|---|---|---|
| `single`         | 1   | `0`                            | Strict 1-GPU baseline. The honest "1 GPU does everything" reference. Will OOM on `xl`. |
| `pp` *(default)* | 1   | `STUDENT_GPUS,TEACHER_GPUS`    | Single process with all GPUs visible. HF's `device_map='auto'` decides placement and ends up doing **implicit pipeline parallelism** under memory pressure. "Naive HF setup, no extra effort." |
| `ddp`            | N\* | `STUDENT_GPUS,TEACHER_GPUS`    | N data-parallel replicas, **one rank per GPU**, full student+teacher copy on each. By default the harness scales per-rank batch down so global generated-token work per step matches the single/remote profiles. |

\*N defaults to the number of GPUs in `STUDENT_GPUS,TEACHER_GPUS` (4 by default).

### Examples

```bash
# Default: pp (single process, multi-GPU pipeline placement under pressure)
bash scripts/bench/run_bench.sh baseline stress
bash scripts/bench/run_bench.sh remote   stress

# Strict 1-GPU baseline — what most "1 GPU vs N GPU" reports actually mean
BASELINE_STRATEGY=single bash scripts/bench/run_bench.sh baseline tiny

# Token-normalized DDP baseline at 4 ranks × 4 GPUs vs the asymmetric remote layout
BASELINE_STRATEGY=ddp    bash scripts/bench/run_bench.sh baseline stress
bash scripts/bench/run_bench.sh remote stress

# Old "same batch on every rank" DDP baseline: higher throughput, more work per step
BASELINE_STRATEGY=ddp BASELINE_DDP_TOKEN_MODE=same_per_rank \
  bash scripts/bench/run_bench.sh baseline stress

# 8-GPU box, asymmetric remote split (DDP baseline auto-scales to 8 ranks)
STUDENT_GPUS=0,1 TEACHER_GPUS=2,3,4,5,6,7 \
  BASELINE_STRATEGY=ddp bash scripts/bench/run_bench.sh baseline stress
STUDENT_GPUS=0,1 TEACHER_GPUS=2,3,4,5,6,7 NUM_PROCESSES=2 \
  bash scripts/bench/run_bench.sh remote stress
```

The baseline run dir is named `baseline-<strategy>__<profile>__<ts>/` so the
three baselines are easy to glob / compare:

```bash
python scripts/bench/summarize.py outputs/bench/baseline-single__stress__* \
                                  outputs/bench/baseline-pp__stress__*     \
                                  outputs/bench/baseline-ddp-global__stress__* \
                                  outputs/bench/baseline-ddp-rank__stress__* \
                                  outputs/bench/remote__stress__*
```

### Sanity checks

- For `BASELINE_STRATEGY=ddp`, the harness verifies `BASELINE_GPUS` lists at
  least `NUM_PROCESSES` GPUs and aborts with a clear error otherwise.
- For `BASELINE_STRATEGY=ddp`, `BASELINE_DDP_TOKEN_MODE=same_global` is the
  default. It divides `PER_DEVICE_TRAIN_BATCH_SIZE` by `NUM_PROCESSES`, so
  `per_device_train_batch_size × num_generations × max_completion_length ×
  num_processes` matches the single/remote profile. For the default 4-GPU
  `stress` and `xl` profiles, this changes per-rank batch from `4` to `1`.
  If the batch cannot be divided evenly, the harness aborts and asks you to
  choose fewer ranks, a larger batch, or `BASELINE_DDP_TOKEN_MODE=same_per_rank`.
- For `BASELINE_STRATEGY=ddp`, the harness also passes
  `--gradient_checkpointing_kwargs '{"use_reentrant": false}'` and
  `--ddp_find_unused_parameters false`. Reentrant checkpointing over LoRA
  adapters can make PyTorch DDP mark the same parameter ready twice during
  backward.
- For DDP baseline, each rank holds a **full copy** of the teacher (~14 GiB in
  bf16 for Qwen2.5-7B). On 4×80G that's fine; on smaller cards check the
  `mem_used_peak_GiB` column before scaling further.
- Set `BASELINE_DDP_TOKEN_MODE=same_per_rank` only when you specifically want a
  max-throughput DDP number where global generated-token work grows with rank
  count.

---

## 4. Monitoring details

`monitor_gpus.sh` polls `nvidia-smi` every `GPU_POLL_INTERVAL_S` seconds (default
`1`) and writes one CSV row per GPU per sample. Columns:

```
epoch_s, iso_ts, index, util_gpu_pct, util_mem_pct,
mem_used_mib, mem_total_mib, power_w, temp_c, sm_clock_mhz
```

The monitor process is spawned in the background by `run_bench.sh` and killed
on script exit (including on Ctrl-C / failure) by the cleanup trap.

If you want per-process attribution rather than per-device, run a second
monitor in another terminal during the bench:

```bash
nvidia-smi pmon -i 0,1,2,3 -d 1 -s u -o T -f outputs/bench/<run>/gpu_pmon.txt
```

(Not wired into the harness automatically because `pmon` requires root on
some driver versions.)

---

## 5. What `summarize.py` reports

Three blocks per side-by-side comparison:

1. **Run config** — the values from `config.env` so you can verify the runs
   are comparable.

2. **Walltime** —
   - `script_total_s` — the entire `run_bench.sh` invocation.
   - `pre_trainer_s` — time spent before the trainer started (model download,
     teacher server boot for remote mode).
   - `trainer_s`     — wall time of the trainer subprocess.
   - `trainer_steps` — last optimizer step seen in the tqdm log.
   - `sec_per_step`  — derived from the last tqdm rate (`s/it` or `1/(it/s)`).

3. **Aggregate GPU stats** (sum over all monitored GPUs):
   - `util_avg_pct`, `util_peak_pct` — utilization. The "mean of GPUs" is the
     unweighted average of per-GPU averages, so it represents headline
     "how busy are my GPUs on average".
   - `mem_used_avg_GiB`, `mem_used_peak_GiB` — summed across GPUs.
   - `power_avg_W`, `power_peak_W`, `energy_Wh` — energy is time-integrated
     `power_w * dt` summed across GPUs. A useful **cost** proxy when the two
     runs use different GPU counts.
   - `gpu_seconds_busy` — `sum(util_pct/100 * dt)` summed across GPUs. This is
     the most useful normalization for "how much actual GPU work did this
     run consume", independent of GPU count or wall time.

4. **Per-GPU detail** — same metrics per device. Useful for spotting that the
   teacher's `device_map='auto'` placement skewed memory onto one GPU, etc.

### Output formats

`summarize.py` always writes a wide-format CSV alongside the runs, in addition
to the printed table. By default the file lands at
`<common-parent>/summary__<timestamp>.csv` (typically `outputs/bench/`).

| Flag | Behavior |
|---|---|
| (none)            | Print the table **and** auto-write `summary__<TS>.csv` next to the runs. |
| `--csv path.csv`  | Write the CSV to a specific path instead of the default. |
| `--no-csv`        | Skip the CSV; only print the table. |
| `--json`          | Print everything as JSON to stdout (the table is suppressed). |

The wide CSV has one row per run and these column groups:

- `run_dir`, `run_name`
- `cfg_*`              — every key from `config.env`
- `wall_*`             — `script_total_s`, `pre_trainer_s`, `trainer_s`,
                         `teacher_boot_s`, `trainer_pure_s`
- `trainer_steps`, `sec_per_step`
- `tokens_generated_per_step`, `tokens_generated_per_sec`
                       — derived; useful for normalizing across profiles
- `agg_*`              — sum/mean/peak across all monitored GPUs
- `gpuN_*` (per device) — util %, mem GiB, power W, gpu_seconds_busy

That makes a clean drop-in for Excel / pandas:

```python
import pandas as pd
df = pd.read_csv("outputs/bench/summary__20260503-141723.csv")
df.pivot_table(index="cfg_PROFILE", columns="cfg_MODE",
               values=["sec_per_step", "agg_mem_used_peak_gib", "agg_gpu_seconds_busy"])
```

### HTML visualization

Use `visualize.py` to turn a summary CSV into a standalone HTML report with
tables and inline SVG charts:

```bash
python scripts/bench/visualize.py outputs/bench/summary__20260503-164642.csv
```

This writes `outputs/bench/summary__20260503-164642.html`. To visualize the
newest summary CSV:

```bash
scripts/bench/visualize_latest.sh
```

The report averages repeated runs by profile and run type, treats older
blank-strategy baseline rows as `baseline-single`, and shows DDP separately
because it changes generated-token work per step when `NUM_PROCESSES > 1`.

---

## 6. Suggested comparison protocol

For a clean comparison:

1. **Pre-cache HF assets**. `Qwen2.5-7B` takes minutes to download on first
   use; you don't want that latency in your `pre_trainer_s` metric.
   ```bash
   huggingface-cli download Qwen/Qwen2.5-0.5B
   huggingface-cli download Qwen/Qwen2.5-7B
   ```

2. **Quiet the box**. Make sure no other CUDA workloads are running. The
   bench reads `nvidia-smi` per-device, not per-process, so background work
   will show up as utilization.

3. **Warm-up run**. Always discard the first run of the day; CUDA kernel JIT,
   filesystem cache and HF tokenizer locks all show up as `pre_trainer_s`
   noise the first time.

4. **Run each mode 2-3 times** with `RUN_TAG=run1`, `RUN_TAG=run2` to
   estimate variance.

5. **Compare**:
   ```bash
   python scripts/bench/summarize.py \
       outputs/bench/baseline__*__run2 \
       outputs/bench/remote__*__run2 \
       --csv outputs/bench/compare.csv
   ```

The KPI to watch first is `gpu_seconds_busy` and `trainer_s` together: if the
remote layout reduces `trainer_s` while keeping `gpu_seconds_busy` similar,
you're benefiting from overlap. If both shrink, you're winning even on raw
hardware utilization (less stalling).

---

## 7. Caveats

- **Accelerate config: bench defaults to single-GPU / multi-GPU, NOT
  DeepSpeed ZeRO-3.** The real run scripts default to
  `examples/accelerate_configs/deepspeed_zero3.yaml`, but with PEFT/LoRA +
  `generate()` ZeRO-3 hits a known crash inside
  `deepspeed/runtime/zero/partitioned_param_coordinator.py` (`IndexError:
  pop from an empty deque`) right after the first rollout. The
  parameter-prefetch trace gets out of sync because
  `unwrap_model_for_generation` calls `summon_full_params`. The bench
  orchestrator therefore sets `ACCELERATE_CONFIG` to `single_gpu.yaml`
  (`NUM_PROCESSES=1`) or `multi_gpu.yaml` (`>1`). Override
  `ACCELERATE_CONFIG=examples/accelerate_configs/deepspeed_zero3.yaml` only
  if you specifically want to benchmark ZeRO-3, and be aware you'll need to
  also fix the DeepSpeed bug (e.g., set `--ds3_gather_for_generation false`
  or switch to FSDP).
- **Tiny data is noisy**. With `MAX_STEPS=4` you mostly measure the first
  generation pass, not steady-state training. For steady-state numbers, set
  `MAX_STEPS=20+` and run for ~10 minutes.
- **Eval is disabled** (`--eval_strategy no`). If you want to bench eval too,
  pass `--eval_strategy steps --eval_steps 5` after the script name.
- **The HTTP path for the remote teacher serializes requests with a single
  global lock** (V1 design). With `NUM_PROCESSES>1` student ranks contend on
  this lock; that's expected for V1 and is part of the "remote" cost.
- **`device_map='auto'` distributes the teacher unevenly** across its GPUs.
  This is normal; see the per-GPU table.
