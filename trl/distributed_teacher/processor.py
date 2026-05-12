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
RemoteLogitsFusionProcessor — drop-in replacement for the in-process
`LogitsFusionProcessor` used by the GRPO trainer (see grpo_trainer.py:122).

It is plug-compatible with `transformers.LogitsProcessorList`: it has the same
`__call__(input_ids, scores) -> scores` signature.

Behavior
--------
- On the very first `__call__` after `reset()` (or construction), the processor
  reads `input_ids` as the full prompt + any already-generated tokens that HF
  may have produced. It opens a new session on the teacher server, runs
  `prefill()` over `input_ids`, and uses the resulting last-step logits to fuse
  with `scores`.
- On every subsequent `__call__`, only `input_ids[:, -1:]` is sent to the
  teacher via `decode_step()`. The teacher reuses its KV cache.
- `reset()` ends the current session so the next `__call__` re-prefills.

The processor *must* be reset between rollouts. The integration in
`grpo_trainer.py` does this via a try/finally around `generate()`.
"""

from __future__ import annotations

import logging

import torch
from transformers.generation.logits_process import LogitsProcessor

from .client import RemoteTeacherClient


logger = logging.getLogger(__name__)


class RemoteLogitsFusionProcessor(LogitsProcessor):
    """
    Args:
        client: a `RemoteTeacherClient` already connected to the teacher server.
        alpha: scalar fusion weight in [0, 1]. `fused = alpha*teacher + (1-alpha)*student`.
        pad_token_id: padding id used to derive the teacher's attention mask
            from `input_ids`.
        alpha_scales: optional [batch] tensor of per-sample multipliers on
            `alpha` (matches the in-process `LogitsFusionProcessor` API).
    """

    def __init__(
        self,
        client: RemoteTeacherClient,
        alpha: float,
        pad_token_id: int,
        alpha_scales: torch.Tensor | None = None,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("`alpha` for logit fusion must be between 0.0 and 1.0.")
        self.client = client
        self.alpha = float(alpha)
        self.pad_token_id = int(pad_token_id)
        self.alpha_scales = alpha_scales
        self._session_started = False

    # ---------------------------------------------------------- lifecycle
    def reset(self) -> None:
        """End the current teacher session if any. Safe to call repeatedly."""
        if self._session_started:
            try:
                self.client.end_session()
            finally:
                self._session_started = False

    def __del__(self):
        try:
            self.reset()
        except Exception:
            pass

    # ---------------------------------------------------------- main hook
    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, seq_len = input_ids.shape

        # --- per-sample alpha (replicating the in-process processor API) ---
        if self.alpha_scales is not None:
            if self.alpha_scales.shape[0] != batch_size:
                raise ValueError(
                    f"alpha_scales batch size {self.alpha_scales.shape[0]} does not match "
                    f"input batch size {batch_size}."
                )
            alpha_tensor = self.alpha_scales.to(dtype=scores.dtype, device=scores.device)
            alpha_tensor = (alpha_tensor * self.alpha).clamp(0.0, 1.0)
            if torch.all(alpha_tensor <= 0.0):
                return scores
        else:
            alpha_tensor = None

        # --- get teacher last-step logits via remote engine ---
        if not self._session_started:
            self.client.start_session(batch_size=batch_size, pad_token_id=self.pad_token_id)
            self._session_started = True
            attention_mask = (input_ids != self.pad_token_id).long()
            teacher_logits = self.client.prefill(input_ids, attention_mask=attention_mask)
        else:
            last_token_ids = input_ids[:, -1:]
            # Build the full attention mask so the teacher's positional bookkeeping
            # remains consistent with left-padded prompts.
            full_attention_mask = (input_ids != self.pad_token_id).long()
            teacher_logits = self.client.decode_step(
                last_token_ids,
                full_attention_mask=full_attention_mask,
            )

        teacher_logits = teacher_logits.to(dtype=scores.dtype, device=scores.device)
        if teacher_logits.size(1) != scores.size(1):
            if teacher_logits.size(1) < scores.size(1):
                raise ValueError(
                    f"Teacher logits vocab size {teacher_logits.size(1)} is smaller than student vocab size "
                    f"{scores.size(1)}. Teacher and student must share a tokenizer."
                )
            teacher_logits = teacher_logits[:, : scores.size(1)]

        if alpha_tensor is None:
            fused = self.alpha * teacher_logits + (1.0 - self.alpha) * scores
        else:
            alpha_tensor = alpha_tensor.view(-1, 1)
            fused = alpha_tensor * teacher_logits + (1.0 - alpha_tensor) * scores
        return fused
