# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CLI entry point: launches a `TeacherInferenceEngine` HTTP server.

Example
-------
On an 8-GPU node, dedicating GPUs 2-7 to the teacher (6 GPUs, naive pipeline
parallel via `device_map='auto'`):

    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \\
        python -m trl.scripts.teacher_serve \\
            --teacher-model meta-llama/Meta-Llama-3-70B-Instruct \\
            --port 8765 \\
            --dtype bfloat16

Then on GPUs 0-1, run the trainer with:

    CUDA_VISIBLE_DEVICES=0,1 accelerate launch \\
        --num_processes 2 \\
        my_grpo_train.py \\
        ... \\
        --teacher_inference_mode remote \\
        --teacher_server_url http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch


logger = logging.getLogger(__name__)


def _parse_dtype(name: str) -> torch.dtype | str:
    if name in {"auto", None}:
        return "auto"
    if not hasattr(torch, name):
        raise ValueError(f"Unknown dtype {name!r}. Try one of: bfloat16, float16, float32.")
    candidate = getattr(torch, name)
    if not isinstance(candidate, torch.dtype):
        raise ValueError(f"{name!r} is not a torch.dtype.")
    return candidate


def _parse_max_memory(spec: str | None) -> dict | None:
    """
    Parse `--max-memory '0=20GiB,1=20GiB,cpu=50GiB'` into {0: '20GiB', ...}.
    Keys that are integers are converted; 'cpu' stays as a string.
    """
    if not spec:
        return None
    out: dict = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        key, _, value = chunk.partition("=")
        key = key.strip()
        value = value.strip()
        if not value:
            raise ValueError(f"Bad --max-memory spec: {chunk!r}")
        if key.isdigit():
            key = int(key)
        out[key] = value
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="trl.scripts.teacher_serve",
        description="Launch a logit-fusion teacher inference server.",
    )
    parser.add_argument(
        "--teacher-model",
        required=True,
        help="HuggingFace model id or local path for the teacher.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port.")
    parser.add_argument(
        "--parallel-strategy",
        default="device_map_auto",
        choices=["device_map_auto", "single_gpu"],
        help="How to place the teacher across the visible GPUs.",
    )
    parser.add_argument(
        "--single-gpu-device",
        default="0",
        help="Device id for parallel-strategy=single_gpu (e.g., '0' or 'cuda:0').",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Model dtype: 'auto', 'bfloat16', 'float16', 'float32'.",
    )
    parser.add_argument(
        "--max-memory",
        default=None,
        help=(
            "Optional max-memory map for device_map='auto', e.g. "
            "'0=20GiB,1=20GiB,2=20GiB,3=20GiB,4=20GiB,5=20GiB'."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to from_pretrained.",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Import lazily so `--help` works without installing torch's heavy deps.
    from ..distributed_teacher.engine import TeacherInferenceEngine
    from ..distributed_teacher.server import serve

    single_gpu_device: str | int = args.single_gpu_device
    if isinstance(single_gpu_device, str) and single_gpu_device.isdigit():
        single_gpu_device = int(single_gpu_device)

    model_kwargs: dict = {}
    if args.trust_remote_code:
        model_kwargs["trust_remote_code"] = True

    logger.warning("Loading teacher model %s...", args.teacher_model)
    engine = TeacherInferenceEngine(
        model_name_or_path=args.teacher_model,
        parallel_strategy=args.parallel_strategy,
        dtype=_parse_dtype(args.dtype),
        single_gpu_device=single_gpu_device,
        max_memory=_parse_max_memory(args.max_memory),
        model_kwargs=model_kwargs,
    )
    logger.warning(
        "Teacher loaded. vocab_size=%s, dtype=%s, parallel=%s.",
        engine.vocab_size,
        engine.model_dtype,
        engine.parallel_strategy,
    )

    serve(engine, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
