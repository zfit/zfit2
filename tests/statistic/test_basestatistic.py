"""Tests for the base statistic class."""

from __future__ import annotations

import pytest

from zfit2.statistic import BaseStatistic


def test_base_statistic_abstract():
    """Test that BaseStatistic is abstract and cannot be instantiated."""
    # Cannot instantiate abstract class
    with pytest.raises(TypeError):
        BaseStatistic(name="test")

    # Should be able to create a concrete subclass
    class DummyStatistic(BaseStatistic):
        def value(self, *args, **kwargs):
            return 1.0

    dummy = DummyStatistic(name="dummy", label="Dummy Test")
    assert dummy.name == "dummy"
    assert dummy.label == "Dummy Test"

    # Test __call__ method (which calls value)
    result = dummy()
    assert result == 1.0

    # Test value method directly
    result = dummy.value()
    assert result == 1.0

    # Test string representations
    assert repr(dummy) == "DummyStatistic(name='dummy', label='Dummy Test')"
    assert str(dummy) == "Dummy Test"


def test_base_statistic_with_default_label():
    """Test BaseStatistic with default label."""

    class SimpleStatistic(BaseStatistic):
        def value(self, *args, **kwargs):
            return 42.0

    stat = SimpleStatistic(name="simple_test")
    assert stat.name == "simple_test"
    assert stat.label == "simple_test"  # Should default to name

    result = stat()
    assert result == 42.0
