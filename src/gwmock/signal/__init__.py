"""Initialization for the signal module."""

from __future__ import annotations

from gwmock.signal.base import SignalSimulator
from gwmock.signal.cbc import CBCSignalSimulator

__all__ = [
    "CBCSignalSimulator",
    "SignalSimulator",
]
