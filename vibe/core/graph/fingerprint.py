"""Merkle recipe fingerprinting for graph nodes.

``fingerprint(node) = H(op_type ‖ canonical(params) ‖ fingerprints of inputs)`` — a
node is keyed by its *recipe*, never its output, so the scheme stays sound even when a
backend is non-deterministic. Change one param or one upstream input and only that node's
transitive dependents get new fingerprints.

Reuses :func:`vibe.core.config.fingerprint.create_dict_fingerprint` (canonical JSON +
SHA-256), so key ordering never affects the hash.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from vibe.core.config.fingerprint import create_dict_fingerprint
from vibe.core.graph.model import Graph, NodeId


def content_hash(path: str | Path) -> str:
    """SHA-256 of a file's bytes.

    Used to lift external file content into a source node's params so that editing the
    file changes the node's fingerprint (the purity discipline made concrete). Source
    nodes are the one place fingerprinting must touch the filesystem; a stat-based
    (mtime+size) fast path is a possible later optimization.

    ``~`` is expanded (``expanduser``) so a home-relative path hashes the same file the
    reading operator opens — the loaders (``read_csv`` etc.) expand it too, keeping the
    fingerprint and the read in lockstep.
    """
    return hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest()


def fingerprint_node(
    graph: Graph, node_id: NodeId, memo: dict[NodeId, str] | None = None
) -> str:
    """Return the recipe fingerprint of ``node_id``, memoized over the DAG.

    Assumes an acyclic graph: recursion has no cycle guard, so a cyclic graph would
    recurse until ``RecursionError``. ``executor.execute`` runs ``validate`` (which
    rejects cycles) first, so this is safe there; call ``validate`` before using this
    helper directly on an untrusted graph.
    """
    memo = {} if memo is None else memo
    if node_id in memo:
        return memo[node_id]

    node = graph.nodes[node_id]
    fp = create_dict_fingerprint(
        {
            "op": node.op,
            "params": node.params,
            "inputs": {
                port: fingerprint_node(graph, dep, memo)
                for port, dep in node.inputs.items()
            },
        }
    )
    memo[node_id] = fp
    return fp
