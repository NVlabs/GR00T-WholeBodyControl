"""Lightweight client for an Isaac-GR00T ZMQ PolicyServer.

The real-robot inference process only needs the PolicyServer wire protocol.  It
does not need the model, Torch, Transformers, or the rest of Isaac-GR00T.  This
module intentionally mirrors the server's msgpack/msgpack-numpy protocol so the
client environment can stay small and does not have to clone model demo assets.
"""

from __future__ import annotations

import functools
from typing import Any

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq


class _MsgSerializer:
    """Isaac-GR00T-compatible serializer with object-array rejection."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        default = functools.partial(_MsgSerializer._safe_encode, chain=None)
        return msgpack.packb(data, default=default)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        object_hook = functools.partial(_MsgSerializer._safe_decode, chain=None)
        return msgpack.unpackb(data, object_hook=object_hook, raw=False)

    @staticmethod
    def _safe_encode(obj: Any, chain=None):
        if isinstance(obj, np.ndarray) and obj.dtype.kind == "O":
            raise TypeError(
                "Refusing to encode object-dtype ndarray; convert it to a "
                "concrete numeric dtype before sending"
            )
        return mnp.encode(obj, chain=chain)

    @staticmethod
    def _safe_decode(obj: Any, chain=None):
        if isinstance(obj, dict):
            nd_value = obj.get(b"nd", obj.get("nd"))
            kind_value = obj.get(b"kind", obj.get("kind"))
            if nd_value and kind_value in (b"O", "O"):
                raise ValueError("Refusing to decode object-dtype ndarray payload")
        return mnp.decode(obj, chain=chain)


class PolicyClient:
    """Minimal client API used by ``run_vla_inference.py``."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str | None = None,
    ):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self.context = zmq.Context()
        self.socket = None
        self._closed = False
        self._init_socket()

    def _init_socket(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call_endpoint(
        self,
        endpoint: str,
        data: dict[str, Any] | None = None,
        requires_input: bool = True,
    ) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        try:
            self.socket.send(_MsgSerializer.to_bytes(request))
            message = self.socket.recv()
        except zmq.error.Again:
            self._init_socket()
            raise

        if message == b"ERROR":
            raise RuntimeError(
                "PolicyServer returned ERROR; verify the server model and protocol"
            )
        response = _MsgSerializer.from_bytes(message)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"PolicyServer error: {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()
            return False

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.call_endpoint(
            "get_action", {"observation": observation, "options": options}
        )
        return tuple(response)

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.call_endpoint("reset", {"options": options})

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.socket is not None:
            self.socket.close(linger=0)
        self.context.term()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
