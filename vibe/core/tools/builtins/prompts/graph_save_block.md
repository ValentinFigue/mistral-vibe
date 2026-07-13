Save part of the current graph as a reusable **block** — a named subgraph you (or a later
session) can wire in like any operator.

You pick `nodes` (omit for the whole graph); the interface is derived automatically:

- **Inputs** — any input in your selection that is fed by a node *outside* it becomes a
  block port named `<node_id>_<port>`.
- **Output** — the one node in the selection that nothing else in it consumes. If there are
  several, pass `output` explicitly.
- **Params** — baked in at their current values by default. List `expose_params` as
  `"<node_id>.<key>"` to keep those configurable when the block is reused.

Guidance:

- **Exclude source nodes** and expose the params that should vary, so the block is reusable
  (open ports) instead of baking a one-off path or literal.
- `name` must be snake_case; it can't collide with an operator or a built-in block.
- Saved blocks persist across sessions and show up in the `graph_patch` catalog.
