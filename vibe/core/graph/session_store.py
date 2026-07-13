"""Where a session's working graph lives on disk.

The `graph_patch` tool mirrors the graph it authors to ``<session_dir>/graph/graph.json``;
`graph_save_block` reads it back from there. Both go through :func:`graph_dir` so the
convention stays in one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibe.core.tools.base import InvokeContext


def graph_dir(ctx: InvokeContext | None) -> Path:
    """The per-session directory holding the working graph + its cache.

    Raises ``ValueError`` if no session/scratchpad directory is available.
    """
    base = (ctx.session_dir or ctx.scratchpad_dir) if ctx else None
    if base is None:
        raise ValueError("no session or scratchpad directory available")
    directory = base / "graph"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def graph_json_path(ctx: InvokeContext | None) -> Path:
    """Path to the mirrored working-graph JSON."""
    return graph_dir(ctx) / "graph.json"
