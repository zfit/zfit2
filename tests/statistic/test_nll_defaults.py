"""Test NLL with default parameters."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as scipy_stats

from zfit2.statistic import NLL


def test_nll_defaults():
    """Test NLL with default parameters."""
    dist = scipy_stats.norm(0, 1)
    data = np.array([0.0, 1.0, -1.0])

    # Create NLL with all defaults
    nll = NLL(dist, data)

    # Check defaults
    assert nll.name == "nll"
    assert nll.label == "Negative Log-Likelihood"
    # Options should be StatisticOptions with mean offset and start_value=10000
    offset_config = nll.options.get_offset_config()
    assert offset_config["method"] == "mean"
    assert offset_config["start_value"] == 10000.0

    # Value should be 10000 (mean offset with start_value=10000)
    value = nll.value()
    assert value == pytest.approx(10000.0)


def test_nll_partial_defaults():
    """Test NLL with some parameters specified."""
    dist = scipy_stats.norm(0, 1)
    data = np.array([0.0, 1.0, -1.0])

    # Only name specified
    nll1 = NLL(dist, data, name="custom")
    assert nll1.name == "custom"
    assert nll1.label == "Negative Log-Likelihood"
    assert nll1.value() == pytest.approx(10000.0)

    # Only label specified
    nll2 = NLL(dist, data, label="Custom Label")
    assert nll2.name == "nll"
    assert nll2.label == "Custom Label"
    assert nll2.value() == pytest.approx(10000.0)


def test_nll_explicit_none_offset():
    """Test that explicitly setting no offset works."""
    from zfit2.statistic import NLLOptions

    dist = scipy_stats.norm(0, 1)
    data = np.array([0.0, 1.0, -1.0])

    # Explicitly use no offset
    nll = NLL(dist, data, options=NLLOptions.none())

    # Should return raw NLL
    expected = -np.sum(dist.logpdf(data))
    assert nll.value() == pytest.approx(expected)
