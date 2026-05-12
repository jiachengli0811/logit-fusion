#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run a small-data benchmark of the GRPO training loop in either:
#   - "baseline" mode (co-located teacher + student, current default)
#   - "remote"   mode (asymmetric: separate teacher inference server)
#
# Records: GPU utilization/memory CSV, full trainer log, walltime breakdown,
# and a small JSON config dump. Designed to produce comparable runs that
# `summarize.py` can diff side-by-side.
#
# Usage:
#   bash scripts/bench/run_bench.sh baseline               # default = pp/tiny
#   bash scripts/bench/run_bench.sh remote   stress        # longer completions / bigger batch
#   bash scripts/bench/run_bench.sh remote   xl            # heaviest preset
#   BASELINE_STRATEGY=ddp bash scripts/bench/run_bench.sh baseline stress
#   BASELINE_STRATEGY=single bash scripts/bench/run_bench.sh baseline tiny
#
# Profiles (set load on the trainer; env-var overrides below still win):
#   tiny    (default)  4 steps, bs=2, gen=2, prompt=512,  comp=256
#                      fast smoke test, ~1 minute
#   stress             8 steps, bs=4, gen=4, prompt=512,  comp=1024
#                      longer completions amplify per-token RPC vs co-located
#                      tradeoff; this is where the asymmetric layout wins shine
#   xl                12 steps, bs=4, gen=8, prompt=1024, comp=2048
#                      heaviest sane preset for the 0.5B/7B pair on 4×80G
#
# Knob overrides via env vars (override whatever the profile set):
#   MAX_STEPS
#   PER_DEVICE_TRAIN_BATCH_SIZE
#   PER_DEVICE_EVAL_BATCH_SIZE
#   NUM_GENERATIONS
#   STEPS_PER_GENERATION         default 2
#   GRADIENT_ACCUMULATION_STEPS  default 1
#   MAX_PROMPT_LENGTH
#   MAX_COMPLETION_LENGTH
#   ALPHA_INIT                   default 0.5
#   MODEL_NAME_OR_PATH           default Qwen/Qwen2.5-0.5B
#   TEACHER_MODEL_NAME_OR_PATH   default Qwen/Qwen2.5-7B
#   STUDENT_GPUS                 default 0
#   TEACHER_GPUS                 default 1,2,3       (remote only)
#   BASELINE_STRATEGY            default pp          (single|pp|ddp; baseline only)
#   BASELINE_GPUS                default <auto>      (set by BASELINE_STRATEGY)
#   BASELINE_DDP_TOKEN_MODE      default same_global (same_global|same_per_rank)
#   BASELINE_DDP_GRADIENT_CHECKPOINTING_KWARGS
#                                default {"use_reentrant": false} (ddp only)
#   BASELINE_DDP_FIND_UNUSED_PARAMETERS
#                                default false       (ddp only)
#   NUM_PROCESSES                default <auto>      (set by BASELINE_STRATEGY)
#   GPU_POLL_INTERVAL_S          default 1
#   BENCH_OUT_ROOT               default outputs/bench
#   RUN_TAG                      default <empty>     (appended to run dir name)
#
# Output layout (one dir per run):
#   $BENCH_OUT_ROOT/<MODE>__<timestamp>[__$RUN_TAG]/
#       config.env             # env snapshot
#       gpu_metrics.csv        # nvidia-smi poll
#       walltime.csv           # phase markers
#       trainer.log            # full training stdout/stderr
#       teacher_server.log     # remote-mode only (copied from output dir)
# ---------------------------------------------------------------------------
set -euo pipefail

# --- 1. arg parsing + cd to repo root -------------------------------------
MODE="${1:-}"
PROFILE="${2:-tiny}"
case "${MODE}" in
  baseline|remote) ;;
  *)
    echo "usage: bash $0 baseline|remote [tiny|stress|xl]" >&2
    exit 2
    ;;
esac
case "${PROFILE}" in
  tiny|stress|xl) ;;
  *)
    echo "usage: bash $0 baseline|remote [tiny|stress|xl]" >&2
    echo "       unknown profile '${PROFILE}'" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(dirname "${SCRIPT_DIR}")"
REPO_ROOT="$(dirname "${SCRIPTS_DIR}")"
cd "${REPO_ROOT}"

# --- 2a. profile presets (only fill in unset values) -----------------------
# `: "${VAR:=value}"` assigns iff VAR is unset/empty, so explicit env vars
# from the caller still take priority over the profile.
case "${PROFILE}" in
  tiny)
    : "${MAX_STEPS:=4}"
    : "${PER_DEVICE_TRAIN_BATCH_SIZE:=2}"
    : "${NUM_GENERATIONS:=2}"
    : "${MAX_PROMPT_LENGTH:=512}"
    : "${MAX_COMPLETION_LENGTH:=256}"
    ;;
  stress)
    : "${MAX_STEPS:=8}"
    : "${PER_DEVICE_TRAIN_BATCH_SIZE:=4}"
    : "${NUM_GENERATIONS:=4}"
    : "${MAX_PROMPT_LENGTH:=512}"
    : "${MAX_COMPLETION_LENGTH:=1024}"
    ;;
  xl)
    : "${MAX_STEPS:=12}"
    : "${PER_DEVICE_TRAIN_BATCH_SIZE:=4}"
    : "${NUM_GENERATIONS:=8}"
    : "${MAX_PROMPT_LENGTH:=1024}"
    : "${MAX_COMPLETION_LENGTH:=2048}"
    ;;
esac

# --- 2b. trailing defaults (knobs not touched by any profile) --------------
_per_device_eval_batch_size_was_auto=0
if [[ -z "${PER_DEVICE_EVAL_BATCH_SIZE:-}" ]]; then
  _per_device_eval_batch_size_was_auto=1
fi

MAX_STEPS="${MAX_STEPS}"
export PER_DEVICE_TRAIN_BATCH_SIZE
export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-${PER_DEVICE_TRAIN_BATCH_SIZE}}"
export NUM_GENERATIONS
export STEPS_PER_GENERATION="${STEPS_PER_GENERATION:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export MAX_PROMPT_LENGTH
export MAX_COMPLETION_LENGTH
export ALPHA_INIT="${ALPHA_INIT:-0.5}"
export ALPHA_DECAY_STEPS="${ALPHA_DECAY_STEPS:-1000}"
export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-0.5B}"
export TEACHER_MODEL_NAME_OR_PATH="${TEACHER_MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
export STUDENT_GPUS="${STUDENT_GPUS:-0}"
export TEACHER_GPUS="${TEACHER_GPUS:-1,2,3}"
export REPORT_TO="${REPORT_TO:-none}"
export BASELINE_DDP_TOKEN_MODE="${BASELINE_DDP_TOKEN_MODE:-same_global}"
export BASELINE_DDP_GRADIENT_CHECKPOINTING_KWARGS="${BASELINE_DDP_GRADIENT_CHECKPOINTING_KWARGS:-{\"use_reentrant\": false}}"
export BASELINE_DDP_FIND_UNUSED_PARAMETERS="${BASELINE_DDP_FIND_UNUSED_PARAMETERS:-false}"

# Baseline strategy shortcut — sets NUM_PROCESSES + BASELINE_GPUS together so
# you don't have to remember the right combination.
#
#   single  -> 1 process on 1 GPU. Strict 1-GPU baseline. Will OOM at xl.
#   pp      -> 1 process, can see the union of student+teacher GPUs. HF's
#              device_map='auto' decides placement, which in practice ends up
#              co-located on the first GPU and only spills under pressure.
#              This is the default and matches "what naive HF gives you".
#   ddp     -> N processes, each on one GPU, full student+teacher REPLICATED
#              per rank. N defaults to the count of student+teacher GPUs.
#              This is the "honest production baseline" — what most people
#              would actually run if they had multiple GPUs available.
#
# The remote layout always uses STUDENT_GPUS+TEACHER_GPUS and ignores this.
BASELINE_STRATEGY="${BASELINE_STRATEGY:-pp}"
_all_baseline_gpus="${STUDENT_GPUS},${TEACHER_GPUS}"
_baseline_gpu_count="$(awk -F',' '{print NF}' <<< "${_all_baseline_gpus}")"

case "${BASELINE_STRATEGY}" in
  single)
    : "${BASELINE_GPUS:=0}"
    : "${NUM_PROCESSES:=1}"
    ;;
  pp)
    : "${BASELINE_GPUS:=${_all_baseline_gpus}}"
    : "${NUM_PROCESSES:=1}"
    ;;
  ddp)
    : "${BASELINE_GPUS:=${_all_baseline_gpus}}"
    : "${NUM_PROCESSES:=${_baseline_gpu_count}}"
    ;;
  *)
    echo "[bench] BASELINE_STRATEGY must be one of: single, pp, ddp (got '${BASELINE_STRATEGY}')" >&2
    exit 2
    ;;
esac
export NUM_PROCESSES
export BASELINE_GPUS

# DDP safety check: per-rank GPU count must be >= NUM_PROCESSES, otherwise
# multiple ranks will fight for the same GPU (silent slowdown / OOM).
if [[ "${MODE}" == "baseline" && "${NUM_PROCESSES}" -gt 1 ]]; then
  _visible_count="$(awk -F',' '{print NF}' <<< "${BASELINE_GPUS}")"
  if [[ "${_visible_count}" -lt "${NUM_PROCESSES}" ]]; then
    echo "[bench] ERROR: NUM_PROCESSES=${NUM_PROCESSES} but BASELINE_GPUS='${BASELINE_GPUS}' only has ${_visible_count} GPUs." >&2
    echo "[bench]        Either reduce NUM_PROCESSES, expand BASELINE_GPUS, or pick a different BASELINE_STRATEGY." >&2
    exit 2
  fi
fi

BASELINE_DDP_REFERENCE_PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE}"
BASELINE_DDP_REFERENCE_TOKENS_PER_STEP="$((PER_DEVICE_TRAIN_BATCH_SIZE * NUM_GENERATIONS * MAX_COMPLETION_LENGTH))"

if [[ "${MODE}" == "baseline" && "${BASELINE_STRATEGY}" == "ddp" ]]; then
  case "${BASELINE_DDP_TOKEN_MODE}" in
    same_global)
      if (( PER_DEVICE_TRAIN_BATCH_SIZE % NUM_PROCESSES != 0 )); then
        echo "[bench] ERROR: BASELINE_DDP_TOKEN_MODE=same_global requires PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE} to be divisible by NUM_PROCESSES=${NUM_PROCESSES}." >&2
        echo "[bench]        For this profile, either set NUM_PROCESSES to a divisor of ${PER_DEVICE_TRAIN_BATCH_SIZE}," >&2
        echo "[bench]        override PER_DEVICE_TRAIN_BATCH_SIZE, or use BASELINE_DDP_TOKEN_MODE=same_per_rank." >&2
        exit 2
      fi
      PER_DEVICE_TRAIN_BATCH_SIZE="$((PER_DEVICE_TRAIN_BATCH_SIZE / NUM_PROCESSES))"
      if (( PER_DEVICE_TRAIN_BATCH_SIZE < 1 )); then
        echo "[bench] ERROR: normalized DDP per-rank batch would be < 1." >&2
        echo "[bench]        Use fewer NUM_PROCESSES or BASELINE_DDP_TOKEN_MODE=same_per_rank." >&2
        exit 2
      fi
      export PER_DEVICE_TRAIN_BATCH_SIZE
      if [[ "${_per_device_eval_batch_size_was_auto}" == "1" ]]; then
        export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE}"
      fi
      ;;
    same_per_rank)
      ;;
    *)
      echo "[bench] BASELINE_DDP_TOKEN_MODE must be one of: same_global, same_per_rank (got '${BASELINE_DDP_TOKEN_MODE}')" >&2
      exit 2
      ;;
  esac
fi

GPU_POLL_INTERVAL_S="${GPU_POLL_INTERVAL_S:-1}"
BENCH_OUT_ROOT="${BENCH_OUT_ROOT:-outputs/bench}"
RUN_TAG="${RUN_TAG:-}"

# --- Accelerate config: pick a non-DeepSpeed config by default --------------
# The default for the underlying run scripts is `deepspeed_zero3.yaml`, which
# triggers a known DeepSpeed ZeRO-3 + LoRA + generate() crash:
#   IndexError: pop from an empty deque
#       in deepspeed/runtime/zero/partitioned_param_coordinator.py:217
# (the parameter-trace coordinator is left in an inconsistent state by the
# `summon_full_params` cycle inside `unwrap_model_for_generation`).
#
# For the bench, we don't need ZeRO-3 — Qwen2.5-0.5B fits trivially on one
# GPU. Use `single_gpu.yaml` for NUM_PROCESSES=1 and `multi_gpu.yaml` for >1.
# Override with ACCELERATE_CONFIG=... if you specifically want to bench
# DeepSpeed.
if [[ -z "${ACCELERATE_CONFIG:-}" ]]; then
  if [[ "${NUM_PROCESSES}" == "1" ]]; then
    ACCELERATE_CONFIG="examples/accelerate_configs/single_gpu.yaml"
  else
    ACCELERATE_CONFIG="examples/accelerate_configs/multi_gpu.yaml"
  fi
fi
export ACCELERATE_CONFIG

# --- 3. per-run output dir -------------------------------------------------
TS="$(date +%Y%m%d-%H%M%S)"
# Embed BASELINE_STRATEGY in the run name when MODE=baseline so that
# single/pp/ddp variants are easy to disambiguate from the directory name.
if [[ "${MODE}" == "baseline" ]]; then
  _run_strategy="${BASELINE_STRATEGY}"
  if [[ "${BASELINE_STRATEGY}" == "ddp" ]]; then
    case "${BASELINE_DDP_TOKEN_MODE}" in
      same_global) _run_strategy="ddp-global" ;;
      same_per_rank) _run_strategy="ddp-rank" ;;
    esac
  fi
  RUN_NAME="${MODE}-${_run_strategy}__${PROFILE}__${TS}${RUN_TAG:+__${RUN_TAG}}"
else
  RUN_NAME="${MODE}__${PROFILE}__${TS}${RUN_TAG:+__${RUN_TAG}}"
fi
RUN_DIR="${BENCH_OUT_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

GPU_CSV="${RUN_DIR}/gpu_metrics.csv"
WALL_CSV="${RUN_DIR}/walltime.csv"
TRAIN_LOG="${RUN_DIR}/trainer.log"
CONFIG_ENV="${RUN_DIR}/config.env"

# --- 4. additional trainer flags injected via "$@" forwarding --------------
# We force eval/save off and cap steps; these are forwarded to gspo.py via the
# `$@` pass-through that the underlying scripts already support.
EXTRA_TRAINER_FLAGS=(
  --max_steps "${MAX_STEPS}"
  --eval_strategy no
  --save_strategy no
  --logging_steps 1
)
if [[ "${MODE}" == "baseline" && "${BASELINE_STRATEGY}" == "ddp" ]]; then
  EXTRA_TRAINER_FLAGS+=(
    --gradient_checkpointing_kwargs "${BASELINE_DDP_GRADIENT_CHECKPOINTING_KWARGS}"
    --ddp_find_unused_parameters "${BASELINE_DDP_FIND_UNUSED_PARAMETERS}"
  )
fi

# --- 5. snapshot config ----------------------------------------------------
{
  echo "# bench run config $(date -u +%FT%TZ)"
  echo "MODE=${MODE}"
  echo "PROFILE=${PROFILE}"
  echo "MAX_STEPS=${MAX_STEPS}"
  echo "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
  echo "PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE}"
  echo "NUM_GENERATIONS=${NUM_GENERATIONS}"
  echo "STEPS_PER_GENERATION=${STEPS_PER_GENERATION}"
  echo "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
  echo "MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH}"
  echo "MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH}"
  echo "ALPHA_INIT=${ALPHA_INIT}"
  echo "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
  echo "TEACHER_MODEL_NAME_OR_PATH=${TEACHER_MODEL_NAME_OR_PATH}"
  echo "STUDENT_GPUS=${STUDENT_GPUS}"
  echo "TEACHER_GPUS=${TEACHER_GPUS}"
  echo "BASELINE_GPUS=${BASELINE_GPUS:-<unset:all>}"
  echo "BASELINE_STRATEGY=${BASELINE_STRATEGY}"
  echo "BASELINE_DDP_TOKEN_MODE=${BASELINE_DDP_TOKEN_MODE}"
  echo "BASELINE_DDP_REFERENCE_PER_DEVICE_TRAIN_BATCH_SIZE=${BASELINE_DDP_REFERENCE_PER_DEVICE_TRAIN_BATCH_SIZE}"
  echo "BASELINE_DDP_REFERENCE_TOKENS_PER_STEP=${BASELINE_DDP_REFERENCE_TOKENS_PER_STEP}"
  echo "BASELINE_DDP_GRADIENT_CHECKPOINTING_KWARGS=${BASELINE_DDP_GRADIENT_CHECKPOINTING_KWARGS}"
  echo "BASELINE_DDP_FIND_UNUSED_PARAMETERS=${BASELINE_DDP_FIND_UNUSED_PARAMETERS}"
  echo "NUM_PROCESSES=${NUM_PROCESSES}"
  echo "ACCELERATE_CONFIG=${ACCELERATE_CONFIG}"
  echo "GPU_POLL_INTERVAL_S=${GPU_POLL_INTERVAL_S}"
  echo "RUN_TAG=${RUN_TAG}"
  echo "EXTRA_TRAINER_FLAGS=(${EXTRA_TRAINER_FLAGS[*]})"
} > "${CONFIG_ENV}"

# --- 6. walltime helper ----------------------------------------------------
echo "phase,epoch_s,iso_ts" > "${WALL_CSV}"
mark_phase() {
  local phase="$1"
  printf '%s,%s,%s\n' "${phase}" "$(date +%s)" "$(date -u +%FT%TZ)" >> "${WALL_CSV}"
  echo "[bench] $(date +%T) phase=${phase}"
}

# --- 7. start GPU monitor in background -----------------------------------
mark_phase script_start
echo "[bench] writing to ${RUN_DIR}/"
echo "[bench] starting GPU monitor (poll=${GPU_POLL_INTERVAL_S}s) -> gpu_metrics.csv"
bash "${SCRIPT_DIR}/monitor_gpus.sh" "${GPU_CSV}" "${GPU_POLL_INTERVAL_S}" &
MONITOR_PID=$!

# --- 8. cleanup on exit ----------------------------------------------------
cleanup() {
  local exit_code=$?
  if kill -0 "${MONITOR_PID}" 2>/dev/null; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
  mark_phase script_end
  echo "[bench] done (exit=${exit_code}) -> ${RUN_DIR}/"
}
trap cleanup EXIT INT TERM

# --- 9. run training -------------------------------------------------------
# We want output dir + run name pinned to the bench run, eval/save off, and
# the existing scripts' env-var-driven config honored. Both wrapper scripts
# accept passthrough args after their flag set, so we forward EXTRA_TRAINER_FLAGS.

export OUTPUT_DIR="${RUN_DIR}/training_output"
export RUN_NAME="${RUN_NAME}"

mark_phase trainer_launch
case "${MODE}" in
  baseline)
    bash "${SCRIPTS_DIR}/run_logit_fusion_gspo.sh" "${EXTRA_TRAINER_FLAGS[@]}" \
      2>&1 | tee "${TRAIN_LOG}"
    BASH_RC=${PIPESTATUS[0]}
    ;;
  remote)
    bash "${SCRIPTS_DIR}/run_logit_fusion_gspo_remote.sh" "${EXTRA_TRAINER_FLAGS[@]}" \
      2>&1 | tee "${TRAIN_LOG}"
    BASH_RC=${PIPESTATUS[0]}
    # Copy the teacher_server.log into the run dir for archival (the underlying
    # script writes it under $OUTPUT_DIR which we already pinned).
    if [[ -f "${OUTPUT_DIR}/teacher_server.log" ]]; then
      cp -f "${OUTPUT_DIR}/teacher_server.log" "${RUN_DIR}/teacher_server.log"
    fi
    ;;
esac
mark_phase trainer_exit

if [[ "${BASH_RC}" -ne 0 ]]; then
  echo "[bench] WARNING: trainer wrapper exited with code ${BASH_RC}"
  exit "${BASH_RC}"
fi
