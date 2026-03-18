"""Init file for the serialize module."""

from __future__ import annotations

from gwmock.data.serialize.decoder import Decoder
from gwmock.data.serialize.encoder import Encoder
from gwmock.data.serialize.serializable import JSONSerializable

__all__ = ["Decoder", "Encoder", "JSONSerializable"]
