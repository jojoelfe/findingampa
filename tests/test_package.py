from __future__ import annotations

import importlib.metadata

import findingampa as m


def test_version():
    assert importlib.metadata.version("findingampa") == m.__version__
