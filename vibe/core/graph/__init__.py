"""Métier — an incremental, content-addressed workflow-graph engine (M0 → M1 keystone).

Public surface:

* :func:`operator` / :func:`get_operator` — declare and look up operators.
* :class:`Graph`, :class:`Node`, :class:`Value`, :class:`Report` — the graph data model.
* :func:`fingerprint_node`, :func:`content_hash` — recipe fingerprinting.
* :class:`CacheStore` — the SQLite content-addressed result store.
* :func:`execute` — the incremental, resumable executor.
"""

from __future__ import annotations

from vibe.core.graph.cache import CacheStore
from vibe.core.graph.executor import (
    GraphEvent,
    GraphValidationError,
    PurityError,
    execute,
    validate,
)
from vibe.core.graph.fingerprint import content_hash, fingerprint_node
from vibe.core.graph.model import Graph, Node, NodeId, Patch, Report, Value
from vibe.core.graph.operators import get_operator, is_registered, operator

__all__ = [
    "CacheStore",
    "Graph",
    "GraphEvent",
    "GraphValidationError",
    "Node",
    "NodeId",
    "Patch",
    "PurityError",
    "Report",
    "Value",
    "content_hash",
    "execute",
    "fingerprint_node",
    "get_operator",
    "is_registered",
    "operator",
    "validate",
]
