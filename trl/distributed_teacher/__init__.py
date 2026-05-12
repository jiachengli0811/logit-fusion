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
Asymmetric distributed teacher inference for logit fusion.

This package separates the teacher model into its own inference engine running
on a dedicated, asymmetric subset of GPUs (e.g., 6 of 8 GPUs) so that the
student's training process can run on the remaining GPUs without contending for
teacher compute. Per-token decode is decoupled across the two engines and the
fusion + sampling step is centralized on the student rank-0 process.

Layout (canonical 8-GPU example):

    Student (trainer) process
        - GPUs 0-1 (TP=1 or TP=2)
        - Holds the trainable policy
        - Drives `model.generate()` with `RemoteLogitsFusionProcessor`
        - Sampling happens here (HF `generate`)

    Teacher inference server (separate process, started independently)
        - GPUs 2-7 (e.g., 6-way pipeline-parallel via `device_map='auto'`)
        - Owns teacher model + per-session KV cache
        - Exposes a per-token logits HTTP API:
            POST /init_session     -> session_id
            POST /prefill          (session_id, prompt_ids)            -> last-step logits
            POST /decode_step      (session_id, last_token_ids)        -> next-step logits
            POST /release_session  (session_id)                        -> 200

    The two communicate via HTTP. The wire format for tensors is the raw
    `torch.save` byte stream sent in the request/response body.

Public API:
    - `TeacherInferenceEngine` (engine module)
    - `serve` (server module - launches an HTTP server wrapping the engine)
    - `RemoteTeacherClient` (client module)
    - `RemoteLogitsFusionProcessor` (processor module)
"""

from .engine import TeacherInferenceEngine
from .processor import RemoteLogitsFusionProcessor


__all__ = [
    "TeacherInferenceEngine",
    "RemoteLogitsFusionProcessor",
]
