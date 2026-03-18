"""Contains utility functions for hashing operations."""

from __future__ import annotations

import hashlib
from pathlib import Path


def compute_file_hash(file_path: str | Path, algorithm: str = "sha256") -> str:
    """Compute the hash of a file using the specified algorithm.

    Args:
        file_path: Path to the file.
        algorithm: Hashing algorithm to use (default is 'sha256').

    Returns:
        The computed hash as a hexadecimal string.
    """
    hash_func = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)
    return f"{algorithm}:{hash_func.hexdigest()}"
