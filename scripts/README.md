# Logit-fusion training: baseline vs. asymmetric distributed teacher

This folder contains shell launchers for two flavors of logit-fusion GRPO/GSPO
training:

| Script | Layout | Per-token cost |
|---|---|---|
| `run_logit_fusion_gspo.sh` | **Co-located** student + teacher on the same GPUs in one process | Teacher forward is **serialized** inside the student's `generate()` decode |
| `run_logit_fusion_gspo_remote.sh` | **Asymmetric**: student on one set of GPUs, teacher on another set, two processes | Teacher forward runs on disjoint GPUs and is contacted per-token via HTTP |

Both end up calling `examples/scripts/gspo.py`, which constructs a `GRPOTrainer`.
The difference is *where* the teacher model lives at decode time.

---

## 1. Architecture

### Baseline (co-located)

```
+----------------------------------------------------+
|  GPU 0..N-1   (single Accelerate process per GPU)  |
|                                                    |
|   Student (trainable, FSDP/DeepSpeed)              |
|   Teacher  (frozen, same process)                  |
|                                                    |
|   HF generate() loop:                              |
|     for t in range(max_new_tokens):                |
|       student_forward()   ----+                    |
|       teacher_forward()   ----+--> fuse --> sample |
|                                                    |
|   `LogitsFusionProcessor` (trl/trainer/grpo_trainer.py:122)
|   runs the teacher *inside* the student's decode.  |
+----------------------------------------------------+
```

Latency per generated token = `student_forward + teacher_forward` (sequential,
on the same devices, fighting for SM/HBM bandwidth).

### Asymmetric distributed teacher (`teacher_inference_mode=remote`)

```
+----------------------------+        +----------------------------------+
|  Trainer (student)         |        |  Teacher inference server        |
|  CUDA_VISIBLE_DEVICES=0,1  |        |  CUDA_VISIBLE_DEVICES=2,...,7    |
|                            |        |                                  |
|  HF generate() loop:       |  HTTP  |  TeacherInferenceEngine          |
|    for t in ...:           +-------->    - device_map='auto' across    |
|      student_forward()     | per    |      its visible GPUs            |
|      RemoteLogitsFusion-   | token  |    - per-session KV cache        |
|        Processor:          |        |    - prefill + decode_step API   |
|         POST /prefill or   |        |                                  |
|         POST /decode_step  |        |                                  |
|        ---> teacher logits |        |                                  |
|      fuse --> sample       |        |                                  |
+----------------------------+        +----------------------------------+
```

Per-token cost on the student side is `student_forward + HTTP RTT`; the
teacher forward runs **in parallel on different GPUs**. With NVLink + loopback
HTTP the per-token RTT is dominated by the teacher's own forward time, so end-
to-end you save roughly the smaller of `student_forward` vs `teacher_forward`
per token. With LoRA students and full-precision multi-billion-param teachers
that is most of the wall-clock.

The teacher process owns its own KV cache. The student trainer process is
unchanged from the baseline except that the in-process `LogitsFusionProcessor`
is swapped for a `RemoteLogitsFusionProcessor` that calls the server.

Key files:

- `trl/distributed_teacher/engine.py` — `TeacherInferenceEngine` (model owner).
- `trl/distributed_teacher/server.py` — stdlib `ThreadingHTTPServer` wrapper.
- `trl/distributed_teacher/client.py` — `RemoteTeacherClient`.
- `trl/distributed_teacher/processor.py` — `RemoteLogitsFusionProcessor`.
- `trl/scripts/teacher_serve.py` — CLI entry point.
- `trl/trainer/grpo_trainer.py` — branches on `teacher_inference_mode`.

---

## 2. Environment

Both flavors run inside the `trl` conda env (the one with `transformers`,
`torch`, `accelerate`, `peft`, `datasets`, etc., already installed):

```bash
conda activate trl
```

The remote flavor has no additional dependencies — the server uses Python's
stdlib `http.server` and the client uses the existing `requests` dep.

---

## 3. Running the baseline (co-located)

```bash
cd logit-fusion
bash scripts/run_logit_fusion_gspo.sh
```

The script's defaults (override via env vars):

| Var | Default |
|---|---|
| `MODEL_NAME_OR_PATH` | `Qwen/Qwen2.5-0.5B` |
| `TEACHER_MODEL_NAME_OR_PATH` | `Qwen/Qwen2.5-7B` |
| `OUTPUT_DIR` | `outputs/logit-fusion-gspo` |
| `RUN_NAME` | `logit-fusion-gspo` |
| `ACCELERATE_CONFIG` | `examples/accelerate_configs/deepspeed_zero3.yaml` |
| `NUM_PROCESSES` | `1` |
| `ALPHA_INIT` | `0.5` |
| `ALPHA_DECAY_STEPS` | `5000` |
| `LEARNING_RATE` | `1e-5` |
| `MAX_PROMPT_LENGTH` | `1024` |
| `MAX_COMPLETION_LENGTH` | `2048` |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | `8` |
| `NUM_GENERATIONS` | `8` |
| `STEPS_PER_GENERATION` | `8` |
| `GRADIENT_ACCUMULATION_STEPS` | `2` |
| `REPORT_TO` | `none` |

Example: 4-GPU baseline run with a different output directory:

```bash
NUM_PROCESSES=4 \
OUTPUT_DIR=/scratch/runs/baseline-4gpu \
RUN_NAME=baseline-4gpu \
bash scripts/run_logit_fusion_gspo.sh
```

Flags forwarded after `--` are passed straight to `examples/scripts/gspo.py`,
e.g.:

```bash
bash scripts/run_logit_fusion_gspo.sh -- --difficulty_tier "<4"
```

---

## 4. Running the asymmetric teacher (`parallel_strategy`)

```bash
cd logit-fusion
bash scripts/run_logit_fusion_gspo_remote.sh
```

This launcher does two things in order:

1. **Starts the teacher inference server** in the background on `TEACHER_GPUS`,
   logging to `${OUTPUT_DIR}/teacher_server.log`. It polls `GET /health` and
   blocks until the server is reachable (or `TEACHER_READY_TIMEOUT_S` elapses).
2. **Launches `accelerate` for the trainer** on `STUDENT_GPUS` with
   `--teacher_inference_mode remote --teacher_server_url
   http://${TEACHER_HOST}:${TEACHER_PORT}` and **no `--teacher_model_name_or_path`**.
   The trainer connects to the server and uses `RemoteLogitsFusionProcessor`
   inside HF `generate()`.

The script registers a trap that kills the teacher server on exit (normal,
SIGINT, or SIGTERM).

### Defaults (override via env vars)

| Var | Default | Notes |
|---|---|---|
| `STUDENT_GPUS` | `0` | Comma-list passed via `CUDA_VISIBLE_DEVICES` to the trainer (default sized for a 4-GPU box) |
| `TEACHER_GPUS` | `1,2,3` | Comma-list passed via `CUDA_VISIBLE_DEVICES` to the server (default sized for a 4-GPU box) |
| `NUM_PROCESSES` | `1` | Trainer Accelerate processes (= one per student GPU) |
| `TEACHER_HOST` | `127.0.0.1` | Server bind host |
| `TEACHER_PORT` | `8765` | Server bind port |
| `TEACHER_DTYPE` | `bfloat16` | Teacher load dtype (`auto`, `bfloat16`, `float16`, `float32`) |
| `TEACHER_PARALLEL_STRATEGY` | `device_map_auto` | See [§5](#5-teacher-parallel-strategies) below |
| `TEACHER_LOG_FILE` | `${OUTPUT_DIR}/teacher_server.log` | Path for server stdout/stderr |
| `TEACHER_READY_TIMEOUT_S` | `600` | Max seconds the launcher waits for `GET /health` to succeed |
| `MODEL_NAME_OR_PATH` | `Qwen/Qwen2.5-0.5B` | Student model |
| `TEACHER_MODEL_NAME_OR_PATH` | `Qwen/Qwen2.5-7B` | Teacher model |
| `ALPHA_INIT`, `ALPHA_DECAY_STEPS`, `LEARNING_RATE`, `MAX_PROMPT_LENGTH`, `MAX_COMPLETION_LENGTH`, `PER_DEVICE_TRAIN_BATCH_SIZE`, `PER_DEVICE_EVAL_BATCH_SIZE`, `NUM_GENERATIONS`, `STEPS_PER_GENERATION`, `GRADIENT_ACCUMULATION_STEPS`, `REPORT_TO` | same as baseline | — |

### Examples

#### 4-GPU box (defaults): 1 student + 3 teacher

```bash
bash scripts/run_logit_fusion_gspo_remote.sh
# student: GPU 0 (1 process)
# teacher: GPUs 1-3 (device_map='auto' across 3 GPUs)
```

#### Canonical 8-GPU split: 2 student + 6 teacher

```bash
STUDENT_GPUS=0,1 \
TEACHER_GPUS=2,3,4,5,6,7 \
NUM_PROCESSES=2 \
bash scripts/run_logit_fusion_gspo_remote.sh
```

#### 4 student GPUs (DDP) + 4 teacher GPUs

```bash
STUDENT_GPUS=0,1,2,3 \
TEACHER_GPUS=4,5,6,7 \
NUM_PROCESSES=4 \
bash scripts/run_logit_fusion_gspo_remote.sh
```

All four student ranks call the same teacher server; the server serializes
requests with a global lock (V1). Throughput-bound workloads at small batch
sizes are typically still faster end-to-end than the co-located baseline
because student forwards continue running while one rank is waiting.

#### Smaller teacher on a single GPU

```bash
TEACHER_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-1.5B \
TEACHER_GPUS=2 \
TEACHER_PARALLEL_STRATEGY=single_gpu \
bash scripts/run_logit_fusion_gspo_remote.sh
```

#### Forward additional `gspo.py` flags

Anything after `--` is appended to the trainer command:

```bash
bash scripts/run_logit_fusion_gspo_remote.sh \
  -- --difficulty_tier "<4" --eval_steps 50
```

---

## 5. Teacher parallel strategies

The `TEACHER_PARALLEL_STRATEGY` env var maps to `--parallel-strategy` on
`trl.scripts.teacher_serve`:

| Value | What it does | When to use |
|---|---|---|
| `device_map_auto` (default) | HF `device_map='auto'` — naive **pipeline parallel**: each layer placed on a different GPU; activations flow through them sequentially. | Always works, no extra deps. Optimal for **decode** (one token at a time, layers serialize cheaply). Sub-optimal for prefill on long prompts. |
| `single_gpu` | Loads the entire teacher onto one GPU specified by `--single-gpu-device`. | Small teachers (e.g., ≤7B in bfloat16) when you have one GPU to spare — lowest latency. |

You can also pass a `--max-memory` map to `teacher_serve` to clamp per-GPU
memory budgets (only effective with `device_map_auto`):

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
  python -m trl.scripts.teacher_serve \
    --teacher-model meta-llama/Meta-Llama-3-70B-Instruct \
    --parallel-strategy device_map_auto \
    --max-memory '0=70GiB,1=70GiB,2=70GiB,3=70GiB,4=70GiB,5=70GiB' \
    --port 8765
```

(The integer keys here refer to the *re-indexed* devices visible inside the
process, i.e. the index after `CUDA_VISIBLE_DEVICES`.)

### Future strategies (not yet implemented)

The `parallel_strategy` enum is the extension point. Two natural follow-ups,
both fitting behind the existing engine + HTTP API:

- **`hf_tp`** — HF native tensor parallel (`tp_plan`/DTensor); much faster
  prefill than pipeline parallel.
- **`vllm`** — vLLM-backed teacher with logits exposed via internal sampler
  hooks. Requires patching vLLM's sampler since per-token raw logits with
  external sampling is not a first-class API there.

---

## 6. Driving the teacher server directly (no trainer)

For debugging or for separate benchmarks, you can launch the server by hand:

```bash
conda activate trl
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
  python -m trl.scripts.teacher_serve \
    --teacher-model Qwen/Qwen2.5-7B \
    --port 8765 \
    --dtype bfloat16
```

Then from any process:

```python
from trl.distributed_teacher.client import RemoteTeacherClient
import torch

client = RemoteTeacherClient(base_url="http://127.0.0.1:8765",
                             connection_timeout=60.0)
print(client.vocab_size, client.model_name_or_path)

prompt_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
client.start_session(batch_size=1)
first_logits = client.prefill(prompt_ids)            # [1, vocab]
next_logits = client.decode_step(torch.tensor([[42]])) # [1, vocab]
client.end_session()
client.close()
```

Endpoints (handy with `curl` for liveness checks):

```
GET  /health           -> {"status":"ok","model":...}
GET  /info             -> {"model":..., "vocab_size":..., "dtype":..., "parallel_strategy":...}
POST /init_session     JSON in/out: {"batch_size": N[, "pad_token_id": id]}
POST /release_session  JSON in:    {"session_id": "..."}
POST /prefill          binary in -> binary out (last-step logits)
POST /decode_step      binary in -> binary out (last-step logits)
```

The binary frame format used by `/prefill` and `/decode_step` is documented in
`trl/distributed_teacher/server.py` (top-level docstring).

---

## 7. Diagnostics

| Symptom | Where to look |
|---|---|
| Trainer hangs at `Connecting to teacher server...` | `${OUTPUT_DIR}/teacher_server.log` — usually OOM or model-download in progress |
| `ConnectionError` from `RemoteTeacherClient` | server died; check `kill -0 $TEACHER_PID` and the log |
| `ValueError: Teacher logits vocab size ... is smaller than student vocab size` | student and teacher tokenizers diverge — pin matching tokenizers |
| `RuntimeError: Session ... has already been prefilled` | bug — open an issue. The trainer is supposed to call `reset()` after every `generate()` and it does |
| Slow per-token decode | check `nvidia-smi` on `TEACHER_GPUS`; if utilization is fragmented across pipeline-parallel stages, try `single_gpu` if the teacher fits |

The trainer uses `try/finally` around `generate()` to call
`RemoteLogitsFusionProcessor.reset()`, which releases the teacher session.
Stale sessions on the server time out only when the server is restarted.

---

## 8. Limitations of V1

- **Rollout-only fusion.** The post-rollout logprob path
  (`_get_fused_per_token_logps`, used by `use_fusion_importance_sampling`) is
  not yet supported with a remote teacher. The trainer auto-disables fusion-IS
  with a warning when `teacher_inference_mode=remote`.
- **HTTP transport.** Tensors cross over loopback HTTP using `torch.save`-bytes
  framing. This is fine on a single node (loopback ~5 GiB/s) but is not the
  fast path; an NCCL side-channel mirroring `VLLMClient.init_communicator` is
  the natural next step.
- **Naive teacher parallelism.** `device_map_auto` is pipeline-parallel; a
  real TP path is gated behind future `parallel_strategy` values.
- **Single global lock on the server.** All requests, all sessions, are
  serialized by one mutex. Concurrent student ranks therefore queue at the
  teacher; this is by design for V1 (correctness over throughput) and is the
  first thing to relax once benchmarks justify the work.
