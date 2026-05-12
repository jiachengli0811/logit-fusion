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
TeacherInferenceEngine — owns a teacher LM and exposes a per-token logits API.

The engine is intended to live in a separate process from the student trainer
and to span its own (typically larger) set of GPUs via one of:

  - `parallel_strategy="device_map_auto"` (default): naive pipeline parallel
    via HF `device_map="auto"`. Always works; per-token latency is dominated by
    the slowest GPU since layers execute sequentially across GPUs but for
    decode this is small relative to vocab projection.

  - `parallel_strategy="single_gpu"`: place the entire teacher on one GPU
    (useful for small teachers and sanity tests).

  - Future: `parallel_strategy="hf_tp"` for native HF tensor parallel (`tp_plan`)
    or `parallel_strategy="vllm"` for a vLLM-backed teacher.

State model
-----------
The engine maintains a dict of *sessions*, each owning a KV cache plus the last
input_ids tensor. The HTTP server / inproc caller is expected to:

  1. Call `init_session(batch_size)` to obtain a `session_id`.
  2. Call `prefill(session_id, prompt_ids, attention_mask)` once per session.
     This returns the last-step logits (the same logits the student would see
     for the *first* generation step).
  3. Call `decode_step(session_id, last_token_ids)` for every subsequent
     decode step. KV cache is reused. Returns the new last-step logits.
  4. Call `release_session(session_id)` to free the KV cache.

The engine is single-threaded by design — KV-cache mutations make concurrent
calls within a session unsafe. The HTTP layer above must serialize requests on
a per-session basis (or globally, which is simpler and what the included server
does).

Note: this engine purposely does NOT perform sampling. The student trainer's
HF `generate()` loop owns sampling, fusion, and EOS handling. The engine only
returns raw next-token logits.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel


@dataclass
class _Session:
    batch_size: int
    past_key_values: Any | None
    cached_seq_len: int
    pad_token_id: int
    # Device of the *first* shard (where input_ids live for forward()). With
    # device_map='auto', HF places embeddings on the first shard and routes
    # subsequent layers; we feed inputs to the first shard's device.
    input_device: torch.device


class TeacherInferenceEngine:
    """
    Owns a frozen teacher LM and exposes prefill/decode_step.

    Args:
        model_name_or_path: HF model id or path.
        parallel_strategy: 'device_map_auto' (default) | 'single_gpu'.
        dtype: model dtype, e.g., torch.bfloat16.
        single_gpu_device: used when `parallel_strategy='single_gpu'`.
        max_memory: passed to `from_pretrained` when `device_map='auto'` to
            give the user explicit control over per-GPU memory budgets. Format
            is `{device_id_or_'cpu': '20GiB'}` or `{device_id: int_bytes}`.
        model_kwargs: extra kwargs for `AutoModelForCausalLM.from_pretrained`.
    """

    def __init__(
        self,
        model_name_or_path: str,
        parallel_strategy: str = "device_map_auto",
        dtype: torch.dtype | str = "auto",
        single_gpu_device: str | int = 0,
        max_memory: dict | None = None,
        model_kwargs: dict | None = None,
    ):
        if parallel_strategy not in {"device_map_auto", "single_gpu"}:
            raise ValueError(
                f"Unknown parallel_strategy={parallel_strategy!r}. "
                "Supported: 'device_map_auto', 'single_gpu'."
            )
        self.parallel_strategy = parallel_strategy
        self.model_name_or_path = model_name_or_path

        load_kwargs: dict = dict(model_kwargs or {})
        load_kwargs.setdefault("dtype" if "dtype" in _from_pretrained_signature() else "torch_dtype", dtype)
        if parallel_strategy == "device_map_auto":
            load_kwargs.setdefault("device_map", "auto")
            if max_memory is not None:
                load_kwargs.setdefault("max_memory", max_memory)
        else:
            # 'single_gpu'
            load_kwargs["device_map"] = None

        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
        if parallel_strategy == "single_gpu":
            self.model = self.model.to(single_gpu_device)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True

        # Tokenizer is optional but useful for decoding diagnostics.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        except Exception:
            self.tokenizer = None

        # Resolve the input device for forward() — HF places embeddings on the
        # first shard for device_map='auto'. Trying to locate it generically:
        self._default_input_device = self._infer_input_device()

        # Session table.
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

        # Default pad token id used when caller doesn't provide one.
        self.default_pad_token_id = (
            self.model.config.pad_token_id
            if self.model.config.pad_token_id is not None
            else (self.tokenizer.pad_token_id if self.tokenizer is not None else 0)
        )

    # ----------------------------------------------------------------- helpers
    def _infer_input_device(self) -> torch.device:
        # Walk the module tree, return the device of the first parameter found
        # on the embedding layer if possible; otherwise the first parameter.
        embed = getattr(self.model, "get_input_embeddings", lambda: None)()
        if embed is not None:
            for p in embed.parameters():
                return p.device
        for p in self.model.parameters():
            return p.device
        raise RuntimeError("Could not infer input device for teacher model.")

    @torch.no_grad()
    def _forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None,
        past_key_values: Any | None,
    ) -> tuple[torch.FloatTensor, Any]:
        kwargs: dict = {"input_ids": input_ids, "use_cache": True}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        outputs = self.model(**kwargs)
        # Last-step logits only — caller only needs the next-token distribution.
        next_step_logits = outputs.logits[:, -1, :]
        return next_step_logits, outputs.past_key_values

    # ------------------------------------------------------------------- API
    def init_session(self, batch_size: int, pad_token_id: int | None = None) -> str:
        sid = uuid.uuid4().hex
        with self._lock:
            self._sessions[sid] = _Session(
                batch_size=batch_size,
                past_key_values=None,
                cached_seq_len=0,
                pad_token_id=pad_token_id if pad_token_id is not None else self.default_pad_token_id,
                input_device=self._default_input_device,
            )
        return sid

    def release_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    @torch.no_grad()
    def prefill(
        self,
        session_id: str,
        prompt_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
    ) -> torch.FloatTensor:
        """
        Run the prefill forward over the entire prompt and seed the KV cache.

        Returns the last-step logits, i.e., the distribution over the *first*
        generated token. The caller (student trainer) typically discards these
        because the student also produced its own first-step logits — except
        when the caller is using teacher-only generation, in which case these
        are exactly what's needed.
        """
        session = self._get_session(session_id)
        if session.past_key_values is not None:
            raise RuntimeError(f"Session {session_id} has already been prefilled.")
        if prompt_ids.dim() != 2:
            raise ValueError("prompt_ids must be a 2D tensor [batch, seq_len].")
        if prompt_ids.size(0) != session.batch_size:
            raise ValueError(
                f"prompt_ids batch size ({prompt_ids.size(0)}) does not match session batch size "
                f"({session.batch_size})."
            )
        prompt_ids = prompt_ids.to(session.input_device, non_blocking=True)
        if attention_mask is None:
            attention_mask = (prompt_ids != session.pad_token_id).long()
        attention_mask = attention_mask.to(session.input_device, non_blocking=True)

        last_logits, past = self._forward(prompt_ids, attention_mask, past_key_values=None)
        session.past_key_values = past
        session.cached_seq_len = prompt_ids.size(1)
        return last_logits

    @torch.no_grad()
    def decode_step(
        self,
        session_id: str,
        last_token_ids: torch.LongTensor,
        full_attention_mask: torch.LongTensor | None = None,
    ) -> torch.FloatTensor:
        """
        Single decode step. Feeds only the new tokens (one per row in batch),
        reuses the cached KV state, returns the next-token logits.

        Args:
            last_token_ids: [batch, 1] tensor of the most recently sampled
                tokens to feed into the teacher.
            full_attention_mask: optional [batch, total_seq_len] full
                attention mask (prompt + tokens generated so far + this step).
                If `None`, an all-ones mask of length `cached_seq_len + 1`
                is constructed (assumes no padding mid-sequence, which holds
                for standard left-padded prompt + monotone generation).
        """
        session = self._get_session(session_id)
        if session.past_key_values is None:
            raise RuntimeError(
                f"Session {session_id} has no KV cache yet — call prefill() before decode_step()."
            )
        if last_token_ids.dim() != 2 or last_token_ids.size(1) != 1:
            raise ValueError("last_token_ids must have shape [batch, 1].")
        if last_token_ids.size(0) != session.batch_size:
            raise ValueError(
                f"last_token_ids batch ({last_token_ids.size(0)}) does not match session batch "
                f"({session.batch_size})."
            )

        last_token_ids = last_token_ids.to(session.input_device, non_blocking=True)
        if full_attention_mask is None:
            full_attention_mask = torch.ones(
                (session.batch_size, session.cached_seq_len + 1),
                dtype=torch.long,
                device=session.input_device,
            )
        else:
            full_attention_mask = full_attention_mask.to(session.input_device, non_blocking=True)

        last_logits, past = self._forward(last_token_ids, full_attention_mask, past_key_values=session.past_key_values)
        session.past_key_values = past
        session.cached_seq_len += 1
        return last_logits

    # --------------------------------------------------------------- internal
    def _get_session(self, session_id: str) -> _Session:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown session_id: {session_id}")
        return session

    @property
    def vocab_size(self) -> int:
        return int(self.model.config.vocab_size)

    @property
    def model_dtype(self) -> torch.dtype:
        for p in self.model.parameters():
            return p.dtype
        return torch.float32


def _from_pretrained_signature() -> set[str]:
    import inspect

    return set(inspect.signature(AutoModelForCausalLM.from_pretrained).parameters.keys())
