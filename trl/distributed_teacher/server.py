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
Stdlib HTTP server that wraps a `TeacherInferenceEngine`.

We deliberately avoid FastAPI/Uvicorn to keep this submodule installable on
training boxes without extra dependencies. Performance over a localhost
HTTP/1.1 connection with `keep-alive` is ~0.3-1.0ms per round-trip plus the
tensor body bytes — for a 150K-vocab fp16 logits tensor with batch 64 that's
~19 MiB / call, which transfers in <10ms on loopback. The teacher forward pass
on multi-GPU pipeline-parallel decode dominates, so HTTP overhead is small.

Wire format
-----------
- Control endpoints (init_session, release_session): JSON in / JSON out.
- Tensor endpoints (prefill, decode_step):
    - Request body:
        4 bytes : little-endian uint32 metadata_len
        N bytes : utf-8 JSON metadata
            { "session_id": "...",
              "tensor_keys": ["prompt_ids", "attention_mask"] }
        Then for each tensor key, in order:
            4 bytes : little-endian uint32 tensor_blob_len
            M bytes : torch.save() bytes for that tensor
    - Response body:
            torch.save() bytes for a single fp32 tensor (the next-step logits).
            (CPU tensor — caller will move to its device.)

This lightweight framing avoids JSON-serializing tensor numbers and avoids the
per-byte overhead of base64. It is *not* meant to be a stable public protocol
yet — both ends are in this repo.
"""

from __future__ import annotations

import io
import json
import logging
import os
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from .engine import TeacherInferenceEngine


logger = logging.getLogger(__name__)


# Module-level handle so the request handler can find the engine. Set by
# `serve()` at startup.
_ENGINE: TeacherInferenceEngine | None = None
# Single global mutex serializing all engine operations. Per-session locking
# would allow concurrent sessions but the V1 trainer issues one session per
# generation pass so global serialization is sufficient and simpler.
_ENGINE_LOCK = threading.Lock()


# ---------------------------------------------------------------- wire format
def _pack_tensor_request(metadata: dict, tensors: dict[str, torch.Tensor]) -> bytes:
    """Pack metadata + ordered tensor blobs into the binary request body."""
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


def _unpack_tensor_request(body: bytes) -> tuple[dict, dict[str, torch.Tensor]]:
    if len(body) < 4:
        raise ValueError("request body too short for metadata length prefix.")
    (meta_len,) = struct.unpack("<I", body[:4])
    if len(body) < 4 + meta_len:
        raise ValueError("request body truncated before metadata block.")
    metadata = json.loads(body[4 : 4 + meta_len].decode("utf-8"))
    cursor = 4 + meta_len
    tensors: dict[str, torch.Tensor] = {}
    for key in metadata.get("tensor_keys", []):
        if len(body) < cursor + 4:
            raise ValueError(f"request body truncated before tensor '{key}' length prefix.")
        (blob_len,) = struct.unpack("<I", body[cursor : cursor + 4])
        cursor += 4
        if len(body) < cursor + blob_len:
            raise ValueError(f"request body truncated within tensor '{key}'.")
        blob = body[cursor : cursor + blob_len]
        cursor += blob_len
        tensors[key] = torch.load(io.BytesIO(blob), weights_only=False)
    return metadata, tensors


def _pack_tensor_response(tensor: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(tensor.detach().to("cpu", dtype=torch.float32).contiguous(), buf)
    return buf.getvalue()


# ------------------------------------------------------------- HTTP handler
class _TeacherRequestHandler(BaseHTTPRequestHandler):
    server_version = "TRLDistributedTeacher/1.0"

    # Mute the noisy default logging — uvicorn-style INFO is overkill here.
    def log_message(self, format, *args):  # noqa: A002 - shadow stdlib parameter
        logger.debug("%s - - %s", self.address_string(), format % args)

    # -- helpers ---------------------------------------------------------
    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(length) if length > 0 else b""

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, status: int, body: bytes, content_type: str = "application/octet-stream") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message})

    # -- routes ----------------------------------------------------------
    def do_GET(self):  # noqa: N802 - stdlib API
        if self.path.rstrip("/") in {"/health", ""}:
            self._send_json(200, {"status": "ok", "model": _ENGINE.model_name_or_path if _ENGINE else None})
            return
        if self.path.rstrip("/") == "/info":
            if _ENGINE is None:
                self._send_error_json(503, "Engine not ready.")
                return
            self._send_json(
                200,
                {
                    "model": _ENGINE.model_name_or_path,
                    "vocab_size": _ENGINE.vocab_size,
                    "dtype": str(_ENGINE.model_dtype),
                    "parallel_strategy": _ENGINE.parallel_strategy,
                },
            )
            return
        self._send_error_json(404, f"unknown GET path {self.path!r}")

    def do_POST(self):  # noqa: N802
        if _ENGINE is None:
            self._send_error_json(503, "Engine not ready.")
            return

        path = self.path.rstrip("/")
        try:
            if path == "/init_session":
                self._handle_init_session()
            elif path == "/release_session":
                self._handle_release_session()
            elif path == "/prefill":
                self._handle_prefill()
            elif path == "/decode_step":
                self._handle_decode_step()
            else:
                self._send_error_json(404, f"unknown POST path {self.path!r}")
        except Exception as exc:
            logger.exception("Error handling %s", path)
            self._send_error_json(500, f"{type(exc).__name__}: {exc}")

    def _handle_init_session(self) -> None:
        body = self._read_body()
        payload = json.loads(body or b"{}")
        batch_size = int(payload["batch_size"])
        pad_token_id = payload.get("pad_token_id")
        with _ENGINE_LOCK:
            sid = _ENGINE.init_session(batch_size=batch_size, pad_token_id=pad_token_id)
        self._send_json(200, {"session_id": sid})

    def _handle_release_session(self) -> None:
        body = self._read_body()
        payload = json.loads(body or b"{}")
        sid = payload["session_id"]
        with _ENGINE_LOCK:
            _ENGINE.release_session(sid)
        self._send_json(200, {"ok": True})

    def _handle_prefill(self) -> None:
        body = self._read_body()
        metadata, tensors = _unpack_tensor_request(body)
        sid = metadata["session_id"]
        prompt_ids = tensors["prompt_ids"].long()
        attention_mask = tensors.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.long()
        with _ENGINE_LOCK:
            logits = _ENGINE.prefill(sid, prompt_ids, attention_mask=attention_mask)
        self._send_bytes(200, _pack_tensor_response(logits))

    def _handle_decode_step(self) -> None:
        body = self._read_body()
        metadata, tensors = _unpack_tensor_request(body)
        sid = metadata["session_id"]
        last_token_ids = tensors["last_token_ids"].long()
        full_attention_mask = tensors.get("full_attention_mask")
        if full_attention_mask is not None:
            full_attention_mask = full_attention_mask.long()
        with _ENGINE_LOCK:
            logits = _ENGINE.decode_step(sid, last_token_ids, full_attention_mask=full_attention_mask)
        self._send_bytes(200, _pack_tensor_response(logits))


# ----------------------------------------------------------------- entry
def serve(
    engine: TeacherInferenceEngine,
    host: str = "0.0.0.0",
    port: int = 8765,
) -> None:
    """
    Start the blocking HTTP server. Call once from the teacher-server process.
    """
    global _ENGINE
    _ENGINE = engine

    server = ThreadingHTTPServer((host, port), _TeacherRequestHandler)
    # ThreadingHTTPServer spins a thread per request; combined with the
    # global engine lock this gives us request serialization with low
    # accept-side latency.
    logger.warning(
        "Teacher inference server listening on http://%s:%s (model=%s, parallel=%s, CUDA_VISIBLE_DEVICES=%s)",
        host,
        port,
        engine.model_name_or_path,
        engine.parallel_strategy,
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.warning("Shutting down teacher server (KeyboardInterrupt).")
    finally:
        server.server_close()
