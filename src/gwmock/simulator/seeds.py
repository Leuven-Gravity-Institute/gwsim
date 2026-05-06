"""Deterministic seed derivation helpers."""

from __future__ import annotations

import hashlib


def _encode_seed_part(kind: str, value: str) -> bytes:
    """Encode one typed seed component with explicit length prefixing."""
    encoded_value = value.encode("utf-8")
    return f"{kind}:{len(encoded_value)}:".encode("ascii") + encoded_value


def derive_seed(parent: int, *labels: str | int) -> int:
    """Derive a deterministic substream seed from ``parent`` and ``labels``."""
    payload = bytearray(_encode_seed_part("int", str(int(parent))))
    for label in labels:
        if isinstance(label, str):
            payload.extend(_encode_seed_part("str", label))
            continue
        if isinstance(label, int):
            payload.extend(_encode_seed_part("int", str(label)))
            continue
        raise TypeError("labels must contain only str or int values.")

    return int(hashlib.sha256(payload).hexdigest(), 16) % (2**63)
