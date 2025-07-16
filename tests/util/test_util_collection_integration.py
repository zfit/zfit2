"""Integration tests for collection utilities with other zfit2 components."""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats

from zfit2.statistic import NLL, NLLOptions
from zfit2.util import is_collection, to_collection


class TestCollectionUtilsWithNLL:
    """Test collection utilities integration with NLL class."""

    def test_nll_uses_collection_logic(self):
        """Test that NLL properly handles various input types via collection utils."""
        dist = scipy_stats.norm(0, 1)
        data = np.array([1, 2, 3])

        # Single distribution and data
        nll1 = NLL(
            dist,
            data,
            options=NLLOptions.none(),
            name="nll",
            label="Negative Log-Likelihood",
        )
        assert len(nll1.dists) == 1
        assert len(nll1.data) == 1

        # Already in lists
        nll2 = NLL(
            [dist],
            [data],
            options=NLLOptions.none(),
            name="nll",
            label="Negative Log-Likelihood",
        )
        assert len(nll2.dists) == 1
        assert len(nll2.data) == 1

        # Tuples
        nll3 = NLL(
            (dist,),
            (data,),
            options=NLLOptions.none(),
            name="nll",
            label="Negative Log-Likelihood",
        )
        assert len(nll3.dists) == 1
        assert len(nll3.data) == 1

        # Generators
        dist_gen = (d for d in [dist])
        data_gen = (d for d in [data])
        nll4 = NLL(
            dist_gen,
            data_gen,
            options=NLLOptions.none(),
            name="nll",
            label="Negative Log-Likelihood",
        )
        assert len(nll4.dists) == 1
        assert len(nll4.data) == 1

    def test_collection_utils_consistency(self):
        """Test that collection utils behave consistently with NLL expectations."""
        # Test that arrays are treated as single items
        arr = np.array([1, 2, 3])
        assert is_collection(arr) is False  # Default excludes arrays
        assert to_collection(arr) == (arr,)  # Wraps array as single item

        # Test that strings are treated as single items
        s = "test"
        assert is_collection(s) is False  # Default excludes strings
        assert to_collection(s) == (s,)  # Wraps string as single item

        # This matches NLL behavior where arrays and strings are single data items
        dist = scipy_stats.norm(0, 1)
        nll = NLL(
            dist,
            arr,
            options=NLLOptions.none(),
            name="nll",
            label="Negative Log-Likelihood",
        )  # Array is treated as one dataset
        assert len(nll.data) == 1
        assert np.array_equal(nll.data[0], arr)
