"""Unit tests for the Decoder class."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

from gwsim.data.serialize.decoder import Decoder


class MockSerializable:
    """Mock class for testing Decoder."""

    def __init__(self, value: int):
        self.value = value

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> MockSerializable:
        """Create from JSON dict."""
        return cls(value=data["value"])


class TestDecoder:
    """Test the Decoder class."""

    def test_decodes_known_type(self):
        """Test decoding an object with known __type__."""
        decoder = Decoder()
        obj_dict = {"__type__": "MockSerializable", "value": 42}
        with patch("gwsim.data.serialize.decoder.importlib.import_module") as mock_import:
            mock_module = mock_import.return_value
            mock_module.MockSerializable = MockSerializable
            result = decoder._object_hook(obj_dict)
            assert isinstance(result, MockSerializable)
            assert result.value == 42

    def test_returns_dict_for_unknown_type(self):
        """Test that unknown __type__ returns the dict unchanged."""
        decoder = Decoder()
        obj_dict = {"__type__": "UnknownType", "data": "test"}
        result = decoder._object_hook(obj_dict)
        assert result == obj_dict

    def test_returns_dict_without_type(self):
        """Test that dict without __type__ is returned unchanged."""
        decoder = Decoder()
        obj_dict = {"key": "value"}
        result = decoder._object_hook(obj_dict)
        assert result == obj_dict

    def test_handles_missing_from_json_dict(self):
        """Test that missing from_json_dict method returns dict unchanged."""
        decoder = Decoder()
        obj_dict = {"__type__": "MockSerializable", "value": 42}
        with patch("gwsim.data.serialize.decoder.importlib.import_module") as mock_import:
            mock_module = mock_import.return_value
            mock_module.MockSerializable = object  # Class without from_json_dict
            result = decoder._object_hook(obj_dict)
            assert result == obj_dict

    def test_full_json_load_with_decoder(self):
        """Test full JSON load using Decoder class."""
        data = {"__type__": "MockSerializable", "value": 100}
        json_str = json.dumps(data)
        with patch("gwsim.data.serialize.decoder.importlib.import_module") as mock_import:
            mock_module = mock_import.return_value
            mock_module.MockSerializable = MockSerializable
            result = json.loads(json_str, cls=Decoder)
            assert isinstance(result, MockSerializable)
            assert result.value == 100

    def test_nested_decoding(self):
        """Test decoding nested structures with serializable objects."""
        nested = {
            "metadata": {"version": "1.0"},
            "object": {"__type__": "MockSerializable", "value": 50},
            "list": [1, {"__type__": "MockSerializable", "value": 75}],
        }
        json_str = json.dumps(nested)
        with patch("gwsim.data.serialize.decoder.importlib.import_module") as mock_import:
            mock_module = mock_import.return_value
            mock_module.MockSerializable = MockSerializable
            result = json.loads(json_str, cls=Decoder)
            assert result["metadata"]["version"] == "1.0"
            assert isinstance(result["object"], MockSerializable)
            assert result["object"].value == 50
            assert result["list"][0] == 1
            assert isinstance(result["list"][1], MockSerializable)
            assert result["list"][1].value == 75
