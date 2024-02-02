from __future__ import annotations

import importlib.metadata

import zfit2 as m


def test_version():
    assert importlib.metadata.version("zfit2") == m.__version__
