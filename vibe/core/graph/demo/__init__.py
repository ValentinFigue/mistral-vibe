"""A deterministic data-pipeline demo for the Métier engine.

Importing this package registers the demo operators (via ``@operator`` in
:mod:`vibe.core.graph.demo.pipeline`).
"""

from __future__ import annotations

from vibe.core.graph.demo.pipeline import build_graph, write_fixtures

__all__ = ["build_graph", "write_fixtures"]
