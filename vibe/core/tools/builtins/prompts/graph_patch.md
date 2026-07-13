The workflow you are building is a persistent, typed graph of operations (a DAG). Your only
way to change it is to emit a **patch**: an ordered list of typed edits applied atomically.

A node has an `id`, an `op` (an operator or block name from the catalog), `params` (literal
scalar arguments), and `inputs` (a `port -> node_id` map wiring an upstream node's result
into this node). An argument is an **input** when it is fed by another node, and a **param**
when it is a literal — you decide which by where you put it, never the operator.

Patch ops (each has a `kind` discriminator):

- `add_node` — `{"kind": "add_node", "node": {"id", "op", "params", "inputs"}}`
- `remove_node` — `{"kind": "remove_node", "id"}` (fails if another node still consumes it)
- `set_param` — `{"kind": "set_param", "id", "key", "value"}`
- `connect` — `{"kind": "connect", "id", "port", "source"}` (wire `source`'s result into `id.port`)
- `disconnect` — `{"kind": "disconnect", "id", "port"}`

Rules:

- Reference only operators/blocks shown in the latest result's catalog. A **block** is a
  reusable subgraph used like any operator — prefer wiring a block over re-authoring its
  nodes.
- The patch is validated before running; an invalid graph is rejected and nothing executes.
  Fix the reported problem and re-emit.
- Execution is incremental: only nodes whose inputs changed recompute. The result reports
  which nodes were `fresh` (ran) vs `cached`, plus the terminal `outputs`.
- Build the graph over one or more patches; edit params to iterate cheaply.
