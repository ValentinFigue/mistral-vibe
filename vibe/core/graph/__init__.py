"""Métier — an incremental, content-addressed workflow-graph engine (M0 → M1 keystone).

Public surface:

* :func:`operator` / :func:`get_operator` — declare and look up operators.
* :class:`Graph`, :class:`Node`, :class:`Value`, :class:`Report` — the graph data model.
* :func:`fingerprint_node`, :func:`content_hash` — recipe fingerprinting.
* :class:`CacheStore` — the SQLite content-addressed result store.
* :func:`execute` — the incremental, resumable executor.
"""

from __future__ import annotations

from vibe.core.graph.blocks import (
    BlockDef,
    BlockError,
    block_from_subgraph,
    expand,
    fold_report,
    get_block,
    is_block,
    load_blocks,
    register_block,
    save_block,
)
from vibe.core.graph.cache import CacheStore
from vibe.core.graph.executor import (
    GraphEvent,
    GraphValidationError,
    PurityError,
    execute,
    validate,
)
from vibe.core.graph.fingerprint import content_hash, fingerprint_node
from vibe.core.graph.model import (
    AddNode,
    Connect,
    Disconnect,
    Graph,
    Node,
    NodeId,
    Patch,
    PatchOp,
    RemoveNode,
    Report,
    SetParam,
    Value,
)
from vibe.core.graph.operators import get_operator, is_registered, operator
from vibe.core.graph.patch import PatchError, apply_patch, changed_nodes

__all__ = [
    "AddNode",
    "BlockDef",
    "BlockError",
    "CacheStore",
    "Connect",
    "Disconnect",
    "Graph",
    "GraphEvent",
    "GraphValidationError",
    "Node",
    "NodeId",
    "Patch",
    "PatchError",
    "PatchOp",
    "PurityError",
    "RemoveNode",
    "Report",
    "SetParam",
    "Value",
    "apply_patch",
    "block_from_subgraph",
    "changed_nodes",
    "content_hash",
    "execute",
    "expand",
    "fingerprint_node",
    "fold_report",
    "get_block",
    "get_operator",
    "is_block",
    "is_registered",
    "load_blocks",
    "operator",
    "register_block",
    "save_block",
    "validate",
]
