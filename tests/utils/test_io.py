"""Unit tests for io utility functions."""

from __future__ import annotations

import numpy as np
import pytest

from gwsim.utils.io import get_file_name_from_template


class MockInstance:
    """Mock instance for testing template expansion."""

    def __init__(self, **attrs):
        for key, value in attrs.items():
            setattr(self, key, value)


class TestGetFileNameFromTemplate:
    """Test suite for get_file_name_from_template function."""

    def test_no_placeholders(self):
        """Test template with no placeholders returns as-is."""
        instance = MockInstance()
        template = "static_filename.txt"
        result = get_file_name_from_template(template, instance)
        assert result == "static_filename.txt"

    def test_single_placeholder_non_array(self):
        """Test single non-array placeholder substitution."""
        instance = MockInstance(name="test")
        template = "{{ name }}.txt"
        result = get_file_name_from_template(template, instance)
        assert result == "test.txt"

    def test_single_placeholder_array(self):
        """Test single array placeholder returns list."""
        instance = MockInstance(items=[1, 2, 3])
        template = "file_{{ items }}.txt"
        result = get_file_name_from_template(template, instance)

        assert np.array_equal(result, np.array(["file_1.txt", "file_2.txt", "file_3.txt"]))

    def test_multiple_placeholders_no_arrays(self):
        """Test multiple non-array placeholders."""
        instance = MockInstance(prefix="data", suffix="end")
        template = "{{ prefix }}_{{ suffix }}.txt"
        result = get_file_name_from_template(template, instance)
        assert result == "data_end.txt"

    def test_multiple_placeholders_with_arrays(self):
        """Test multiple placeholders including arrays."""
        instance = MockInstance(prefix="data", numbers=[1, 2], letters=["a", "b"])
        template = "{{ prefix }}_{{ numbers }}_{{ letters }}.txt"
        result = get_file_name_from_template(template, instance)
        expected = [["data_1_a.txt", "data_1_b.txt"], ["data_2_a.txt", "data_2_b.txt"]]
        assert np.array_equal(result, np.array(expected))

    def test_excluded_placeholders(self):
        """Test excluded placeholders are not substituted."""
        instance = MockInstance(name="test", excluded="ignore")
        template = "{{ name }}_{{ excluded }}.txt"
        result = get_file_name_from_template(template, instance, exclude={"excluded"})
        assert result == "test_{{ excluded }}.txt"

    def test_excluded_with_arrays(self):
        """Test excluded array placeholders are not expanded."""
        instance = MockInstance(name="test", excluded=[1, 2])
        template = "{{ name }}_{{ excluded }}.txt"
        result = get_file_name_from_template(template, instance, exclude={"excluded"})
        assert result == "test_{{ excluded }}.txt"

    def test_missing_attribute(self):
        """Test missing attribute raises ValueError."""
        instance = MockInstance(name="test")
        template = "{{ name }}_{{ missing }}.txt"
        with pytest.raises(ValueError, match="Attribute 'missing' not found"):
            get_file_name_from_template(template, instance)

    def test_tuple_as_array(self):
        """Test tuple is treated as array-like."""
        instance = MockInstance(items=(1, 2))
        template = "file_{{ items }}.txt"
        result = get_file_name_from_template(template, instance)
        assert np.array_equal(result, np.array(["file_1.txt", "file_2.txt"]))

    def test_iterable_non_string(self):
        """Test iterable (non-string) is treated as array-like."""
        instance = MockInstance(items=range(3))
        template = "file_{{ items }}.txt"
        result = get_file_name_from_template(template, instance)
        assert np.array_equal(result, np.array(["file_0.txt", "file_1.txt", "file_2.txt"]))

    def test_string_not_treated_as_array(self):
        """Test string is not treated as array-like."""
        instance = MockInstance(name="hello")
        template = "{{ name }}.txt"
        result = get_file_name_from_template(template, instance)
        assert result == "hello.txt"

    def test_empty_template(self):
        """Test empty template."""
        instance = MockInstance()
        template = ""
        result = get_file_name_from_template(template, instance)
        assert result == ""

    def test_duplicate_placeholders(self):
        """Test duplicate placeholders are handled correctly."""
        instance = MockInstance(name="test")
        template = "{{ name }}_{{ name }}.txt"
        result = get_file_name_from_template(template, instance)
        assert result == "test_test.txt"

    def test_mixed_excluded_and_included(self):
        """Test mix of excluded and included placeholders with arrays."""
        instance = MockInstance(included=[1, 2], excluded=["a", "b"])
        template = "{{ included }}_{{ excluded }}.txt"
        result = get_file_name_from_template(template, instance, exclude={"excluded"})
        expected = ["1_{{ excluded }}.txt", "2_{{ excluded }}.txt"]
        assert np.array_equal(result, np.array(expected))
