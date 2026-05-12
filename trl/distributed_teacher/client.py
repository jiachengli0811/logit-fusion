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
Client used by the trainer (student-side) to talk to a `TeacherInferenceEngine`
HTTP server.

Lifecycle (per generation pass):

    client.start_session(batch_size, pad_token_id)
    teacher_logits_first = client.prefill(prompt_ids, attention_mask=...)
    for t in range(max_new_tokens):
        teacher_logits_t = client.decode_step(last_token_ids)
    client.end_session()
"""

from __future__ import annotations

import io
import json
import logging
import struct
import time
from typing import Any
from urllib.parse import urlparse

import torch


logger = logging.getLogger(__name__)


def _is_requests_available() -> bool:
    try:
        import requests  # noqa: F401

        return True
    except ImportError:
        return False


class RemoteTeacherClient:
    """
    Args:
        base_url: e.g., 'http://127.0.0.1:8765'.
        connection_timeout: seconds to wait for the server to come up before
            failing `__init__`. Set to 0 to require the server be up immediately.
        request_timeout: per-call timeout (seconds).
        retries: HTTP retries on connection error per call.
    """

    def __init__(
        self,
        base_url: str,
        connection_timeout: float = 0.0,
        request_timeout: float = 600.0,
        retries: int = 0,
    ):
        if not _is_requests_available():
            raise ImportError(
                "`requests` is not installed. Install it with `pip install requests` to use "
                "RemoteTeacherClient."
            )
        import requests

        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"base_url must start with http(s)://, got {base_url!r}")
        self.base_url = f"{parsed.scheme}://{parsed.netloc}"
        self.session = requests.Session()
        self.request_timeout = request_timeout
        self.retries = max(0, int(retries))

        self._session_id: str | None = None
        self._info: dict[str, Any] | None = None

        self._wait_for_server(connection_timeout)
        self._info = self._get_info()

    # ------------------------------------------------------------ helpers
    def _wait_for_server(self, total_timeout: float, retry_interval: float = 1.0) -> None:
        import requests

        url = f"{self.base_url}/health"
        start = time.time()
        while True:
            try:
                resp = self.session.get(url, timeout=5.0)
                if resp.status_code == 200:
                    logger.info("Teacher server reachable at %s.", self.base_url)
                    return
            except requests.RequestException:
                pass
            if (time.time() - start) >= total_timeout:
                raise ConnectionError(
                    f"Teacher inference server not reachable at {self.base_url} after {total_timeout}s."
                )
            time.sleep(retry_interval)

    def _get_info(self) -> dict[str, Any]:
        resp = self.session.get(f"{self.base_url}/info", timeout=self.request_timeout)
        resp.raise_for_status()
        return resp.json()

    # --------------------------------------- properties exposed for the trainer
    @property
    def vocab_size(self) -> int | None:
        return self._info.get("vocab_size") if self._info else None

    @property
    def model_name_or_path(self) -> str | None:
        return self._info.get("model") if self._info else None

    # ---------------------------------------------------------- session API
    def start_session(self, batch_size: int, pad_token_id: int | None = None) -> str:
        if self._session_id is not None:
            raise RuntimeError("Session already in progress; call end_session() first.")
        payload = {"batch_size": int(batch_size)}
        if pad_token_id is not None:
            payload["pad_token_id"] = int(pad_token_id)
        resp = self.session.post(
            f"{self.base_url}/init_session",
            json=payload,
            timeout=self.request_timeout,
        )
        resp.raise_for_status()
        self._session_id = resp.json()["session_id"]
        return self._session_id

    def end_session(self) -> None:
        if self._session_id is None:
            return
        try:
            self.session.post(
                f"{self.base_url}/release_session",
                json={"session_id": self._session_id},
                timeout=self.request_timeout,
            )
        except Exception:
            logger.warning("Failed to release session %s on the teacher server.", self._session_id, exc_info=True)
        finally:
            self._session_id = None

    # ---------------------------------------------------------- tensor calls
    def prefill(self, prompt_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self._session_id is None:
            raise RuntimeError("No active session — call start_session() first.")
        tensors: dict[str, torch.Tensor] = {"prompt_ids": prompt_ids.detach().to("cpu", dtype=torch.long)}
        if attention_mask is not None:
            tensors["attention_mask"] = attention_mask.detach().to("cpu", dtype=torch.long)
        body = _pack_tensor_request({"session_id": self._session_id}, tensors)
        return self._post_tensor("/prefill", body)

    def decode_step(
        self,
        last_token_ids: torch.Tensor,
        full_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._session_id is None:
            raise RuntimeError("No active session — call start_session() first.")
        tensors: dict[str, torch.Tensor] = {"last_token_ids": last_token_ids.detach().to("cpu", dtype=torch.long)}
        if full_attention_mask is not None:
            tensors["full_attention_mask"] = full_attention_mask.detach().to("cpu", dtype=torch.long)
        body = _pack_tensor_request({"session_id": self._session_id}, tensors)
        return self._post_tensor("/decode_step", body)

    # ----------------------------------------------------------- transport
    def _post_tensor(self, path: str, body: bytes) -> torch.Tensor:
        import requests

        url = f"{self.base_url}{path}"
        last_exc: BaseException | None = None
        attempts = 1 + self.retries
        for attempt in range(attempts):
            try:
                resp = self.session.post(
                    url,
                    data=body,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=self.request_timeout,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"Teacher server {path} returned {resp.status_code}: {resp.text[:500]}")
                return torch.load(io.BytesIO(resp.content), weights_only=False)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                if attempt + 1 < attempts:
                    time.sleep(0.1 * (attempt + 1))
        raise ConnectionError(f"Failed to call {url} after {attempts} attempts: {last_exc}") from last_exc

    # -------------------------------------------------------------- cleanup
    def close(self) -> None:
        try:
            self.end_session()
        finally:
            self.session.close()

    def __enter__(self) -> "RemoteTeacherClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# Re-implementation of the wire packer here to keep client.py importable
# without dragging in the server module (which imports torch + http.server).
def _pack_tensor_request(metadata: dict, tensors: dict[str, torch.Tensor]) -> bytes:
    if "tensor_keys" not in metadata:
        metadata = {**metadata, "tensor_keys": list(tensors.keys())}
    meta_bytes = json.dumps(metadata).encode("utf-8")
    parts: list[bytes] = [struct.pack("<I", len(meta_bytes)), meta_bytes]
    for key in metadata["tensor_keys"]:
        buf = io.BytesIO()
        torch.save(tensors[key].cpu().contiguous(), buf)
        blob = buf.getvalue()
        parts.append(struct.pack("<I", len(blob)))
        parts.append(blob)
    return b"".join(parts)
