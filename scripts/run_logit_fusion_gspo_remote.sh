#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Logit-fusion GSPO with an ASYMMETRIC student/teacher layout.
#
# This launches:
#   1. A teacher inference server on $TEACHER_GPUS (e.g. "2,3,4,5,6,7")
#      using `device_map='auto'` to spread the teacher across those GPUs.
#   2. The student trainer on $STUDENT_GPUS (e.g. "0,1") via Accelerate.
#      It talks to the teacher over HTTP (`teacher_inference_mode=remote`).
#
# Per-token decode runs in parallel: the student does one forward on its own
# GPUs, the teacher does one (KV-cached) forward on its own GPUs, and only
# the next-step logits cross the wire. Sampling lives on the student.
#
# Comparison to the co-located baseline (`run_logit_fusion_gspo.sh`):
#   - baseline: student + teacher contend for the same GPUs in one process,
#     teacher forward is *serialized* inside the student's per-token decode.
#   - remote:   student + teacher run on disjoint GPUs in two processes,
#     teacher forward overlaps with student forward on different GPUs.
# ---------------------------------------------------------------------------
set -euo pipefail

# Run from the repo root regardless of where the script was invoked from, so
# that relative paths like `examples/accelerate_configs/...` and
# `examples/scripts/gspo.py` resolve correctly.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-0.5B}"
TEACHER_MODEL_NAME_OR_PATH="${TEACHER_MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/logit-fusion-gspo-remote}"
RUN_NAME="${RUN_NAME:-logit-fusion-gspo-remote}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-examples/accelerate_configs/deepspeed_zero3.yaml}"

# --- GPU partition ---------------------------------------------------------
# Defaults are sized for a 4-GPU box: 1 student GPU + 3 teacher GPUs.
# For an 8-GPU box, override via env vars, e.g.:
#   STUDENT_GPUS=0,1 TEACHER_GPUS=2,3,4,5,6,7 NUM_PROCESSES=2 \
#       bash scripts/run_logit_fusion_gspo_remote.sh
STUDENT_GPUS="${STUDENT_GPUS:-0}"
TEACHER_GPUS="${TEACHER_GPUS:-1,2,3}"
# NUM_PROCESSES is the number of trainer processes (= len(STUDENT_GPUS) when
# using one rank per GPU on the student side).
NUM_PROCESSES="${NUM_PROCESSES:-1}"

# --- Teacher server --------------------------------------------------------
TEACHER_HOST="${TEACHER_HOST:-127.0.0.1}"
TEACHER_PORT="${TEACHER_PORT:-8765}"
TEACHER_DTYPE="${TEACHER_DTYPE:-bfloat16}"
TEACHER_PARALLEL_STRATEGY="${TEACHER_PARALLEL_STRATEGY:-device_map_auto}"
TEACHER_LOG_LEVEL="${TEACHER_LOG_LEVEL:-info}"
TEACHER_LOG_FILE="${TEACHER_LOG_FILE:-${OUTPUT_DIR}/teacher_server.log}"
TEACHER_READY_TIMEOUT_S="${TEACHER_READY_TIMEOUT_S:-600}"

# --- Training knobs --------------------------------------------------------
ALPHA_INIT="${ALPHA_INIT:-0.5}"
ALPHA_DECAY_STEPS="${ALPHA_DECAY_STEPS:-5000}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-8}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
STEPS_PER_GENERATION="${STEPS_PER_GENERATION:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
REPORT_TO="${REPORT_TO:-none}"

export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TRL_MATH_VERIFY_PARSING_TIMEOUT="${TRL_MATH_VERIFY_PARSING_TIMEOUT:-30}"
export TRL_MATH_VERIFY_VERIFY_TIMEOUT="${TRL_MATH_VERIFY_VERIFY_TIMEOUT:-30}"

mkdir -p "$(dirname "${TEACHER_LOG_FILE}")"

# --- GPU sanity check ------------------------------------------------------
# Confirm every device id requested in STUDENT_GPUS / TEACHER_GPUS actually
# exists on this box and that the two sets are disjoint. Without this check,
# CUDA silently masks down to whatever physical devices exist, which can lead
# to confusing partial runs (e.g. asking for 6 teacher GPUs but only getting 2).
if command -v nvidia-smi >/dev/null 2>&1; then
  NUM_VISIBLE_GPUS="$(nvidia-smi --list-gpus | wc -l)"
else
  NUM_VISIBLE_GPUS=0
fi

_check_gpu_list() {
  local label="$1" csv="$2"
  IFS=',' read -ra _ids <<< "${csv}"
  for id in "${_ids[@]}"; do
    if ! [[ "$id" =~ ^[0-9]+$ ]] || (( id >= NUM_VISIBLE_GPUS )); then
      echo "[run_logit_fusion_gspo_remote] ERROR: ${label}='${csv}' references device ${id}," \
        "but this box has ${NUM_VISIBLE_GPUS} visible GPU(s)."
      echo "  Set ${label} (and the matching counterpart) via env var, e.g.:"
      echo "    STUDENT_GPUS=0 TEACHER_GPUS=1,2,3 NUM_PROCESSES=1 bash $0"
      exit 1
    fi
  done
}

if (( NUM_VISIBLE_GPUS > 0 )); then
  _check_gpu_list STUDENT_GPUS "${STUDENT_GPUS}"
  _check_gpu_list TEACHER_GPUS "${TEACHER_GPUS}"
  # Disjointness check.
  IFS=',' read -ra _s_ids <<< "${STUDENT_GPUS}"
  IFS=',' read -ra _t_ids <<< "${TEACHER_GPUS}"
  for s in "${_s_ids[@]}"; do
    for t in "${_t_ids[@]}"; do
      if [[ "$s" == "$t" ]]; then
        echo "[run_logit_fusion_gspo_remote] ERROR: GPU ${s} appears in both STUDENT_GPUS and TEACHER_GPUS."
        exit 1
      fi
    done
  done
fi
echo "[run_logit_fusion_gspo_remote] GPU plan: student=${STUDENT_GPUS}  teacher=${TEACHER_GPUS}  (visible=${NUM_VISIBLE_GPUS})"

# --- Launch teacher server in background -----------------------------------
TEACHER_PID=""
cleanup() {
  if [[ -n "${TEACHER_PID}" ]] && kill -0 "${TEACHER_PID}" 2>/dev/null; then
    echo "[run_logit_fusion_gspo_remote] stopping teacher server (pid=${TEACHER_PID})..."
    kill "${TEACHER_PID}" 2>/dev/null || true
    # Give it a moment, then force.
    sleep 2
    kill -9 "${TEACHER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[run_logit_fusion_gspo_remote] launching teacher server"
echo "  model=${TEACHER_MODEL_NAME_OR_PATH}"
echo "  GPUs=${TEACHER_GPUS}  strategy=${TEACHER_PARALLEL_STRATEGY}  dtype=${TEACHER_DTYPE}"
echo "  bind=${TEACHER_HOST}:${TEACHER_PORT}  log=${TEACHER_LOG_FILE}"

CUDA_VISIBLE_DEVICES="${TEACHER_GPUS}" \
  python -m trl.scripts.teacher_serve \
    --teacher-model "${TEACHER_MODEL_NAME_OR_PATH}" \
    --host "${TEACHER_HOST}" \
    --port "${TEACHER_PORT}" \
    --dtype "${TEACHER_DTYPE}" \
    --parallel-strategy "${TEACHER_PARALLEL_STRATEGY}" \
    --log-level "${TEACHER_LOG_LEVEL}" \
    >"${TEACHER_LOG_FILE}" 2>&1 &
TEACHER_PID=$!
echo "[run_logit_fusion_gspo_remote] teacher server pid=${TEACHER_PID}"

# Wait for the teacher to come up.
SECONDS=0
until curl -sfo /dev/null "http://${TEACHER_HOST}:${TEACHER_PORT}/health"; do
  if ! kill -0 "${TEACHER_PID}" 2>/dev/null; then
    echo "[run_logit_fusion_gspo_remote] teacher server died during startup; tail of log:"
    tail -n 50 "${TEACHER_LOG_FILE}" || true
    exit 1
  fi
  if (( SECONDS > TEACHER_READY_TIMEOUT_S )); then
    echo "[run_logit_fusion_gspo_remote] teacher server not ready after ${TEACHER_READY_TIMEOUT_S}s; tail of log:"
    tail -n 50 "${TEACHER_LOG_FILE}" || true
    exit 1
  fi
  sleep 2
done
echo "[run_logit_fusion_gspo_remote] teacher ready after ${SECONDS}s."

# --- Launch the trainer ----------------------------------------------------
# Note: we deliberately do NOT pass --teacher_model_name_or_path. The trainer
# branches on --teacher_inference_mode=remote and connects to the server above.

CUDA_VISIBLE_DEVICES="${STUDENT_GPUS}" \
  accelerate launch \
    --num_processes "${NUM_PROCESSES}" \
    --config_file "${ACCELERATE_CONFIG}" \
    examples/scripts/gspo.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --teacher_inference_mode remote \
    --teacher_server_url "http://${TEACHER_HOST}:${TEACHER_PORT}" \
    --use_vllm False \
    --use_vllm_eval False \
    --logit_fusion_alpha "${ALPHA_INIT}" \
    --logit_fusion_alpha_schedule linear \
    --logit_fusion_alpha_decay_steps "${ALPHA_DECAY_STEPS}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --learning_rate "${LEARNING_RATE}" \
    --lr_scheduler_type constant \
    --dtype bfloat16 \
    --max_prompt_length "${MAX_PROMPT_LENGTH}" \
    --max_completion_length "${MAX_COMPLETION_LENGTH}" \
    --use_peft \
    --lora_target_modules q_proj v_proj k_proj o_proj down_proj up_proj gate_proj \
    --log_completions \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --num_generations "${NUM_GENERATIONS}" \
    --steps_per_generation "${STEPS_PER_GENERATION}" \
    --loss_type grpo \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --epsilon 3e-4 \
    --epsilon_high 4e-4 \
    --beta 0.0 \
    --importance_sampling_level sequence \
    --report_to "${REPORT_TO}" \
    --num_completions_to_print 2 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 1000 \
    "$@"
