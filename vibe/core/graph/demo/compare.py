"""Head-to-head: the Métier graph vs. a plain transcript agent — ``python -m vibe.core.graph.demo.compare``.

We run the *same* computation two ways and measure what reaches the model:

* **Transcript baseline** — models the "original version": a linear tool-call agent. Every
  step's full result is inlined into the conversation, and because there is no cache a re-run
  recomputes and re-inlines *everything*. This is a faithful model of context growth over the
  same work, not a claim about any specific competing agent.
* **Graph** — the engine. Intermediate payloads live in the content-addressed cache; the model
  sees only compact handles (``ref:<fp> <Type> — <preview>``) for terminal outputs plus the
  catalog once (exactly what :meth:`GraphPatch.get_llm_content` emits). A re-run recomputes only
  the dirty subgraph and the model-facing context barely moves.

Two headline wins, one table per workflow: **context size** (approx tokens) and **recompute
cost** (nodes computed on re-run). Token counts are ``approx_token_count`` (chars/4), so read
them as estimates, not a specific tokenizer's output.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import random
import tempfile

from vibe.core.graph import render
from vibe.core.graph.cache import CacheStore
from vibe.core.graph.demo.research import build_research_graph as _research_build
from vibe.core.graph.executor import execute
from vibe.core.graph.fingerprint import content_hash
import vibe.core.graph.library.analysis  # noqa: F401  (registers the analysis operator kit)
import vibe.core.graph.library.doc_reviewer  # noqa: F401  (registers the document-review kit)
from vibe.core.graph.model import Graph, Node, Report, Value
from vibe.core.tools.builtins.graph_patch import (
    _MAX_OUTPUT_CHARS,
    GraphPatch,
    GraphPatchConfig,
    GraphPatchResult,
    GraphPatchState,
    OutputHandle,
    _render_catalog,
)
from vibe.core.utils.tokens import approx_token_count


def _tool() -> GraphPatch:
    """A GraphPatch whose only role here is to lend its real ``get_llm_content``."""
    return GraphPatch(config_getter=lambda: GraphPatchConfig(), state=GraphPatchState())


_REGIONS = ("emea", "amer", "apac", "latam")
_COUNTRIES = ("fr", "us", "de", "jp", "br", "in", "gb", "ca")
_ACTIONS = ("view", "signup", "purchase", "refund")


def _write_events_csv(work_dir: Path, n_rows: int = 2000) -> Path:
    """Write a deterministic ~2k-row events CSV (large intermediates for the comparison)."""
    path = work_dir / "events.csv"
    if not path.exists():
        rng = random.Random(7)
        lines = ["event_id,region,country,action,amount"]
        lines += [
            f"{i},{rng.choice(_REGIONS)},{rng.choice(_COUNTRIES)},"
            f"{rng.choice(_ACTIONS)},{round(rng.uniform(1.0, 500.0), 2)}"
            for i in range(n_rows)
        ]
        path.write_text("\n".join(lines) + "\n")
    return path


def build_analytics_graph(work_dir: Path, *, key: str = "country") -> Graph:
    """The analytics workflow on the real kit: read_csv → filter → group_by → top_n → report."""
    path = _write_events_csv(work_dir)
    g = Graph()
    g.add(Node(id="events", op="read_csv",
               params={"path": str(path), "content_fp": content_hash(path)}))
    g.add(Node(id="filtered", op="filter_rows",
               params={"column": "action", "op": "==", "value": "purchase"},
               inputs={"table": "events"}))
    g.add(Node(id="grouped", op="group_by",
               params={"keys": [key], "metric": "amount", "aggs": ["sum", "count"]},
               inputs={"table": "filtered"}))
    g.add(Node(id="ranked", op="top_n", params={"by": "amount_sum", "n": 5},
               inputs={"table": "grouped"}))
    g.add(Node(id="report", op="to_markdown",
               params={"title": "Top markets by purchase revenue", "max_rows": 10},
               inputs={"table": "ranked"}))
    return g


def build_research_graph(work_dir: Path, *, keyword: str = "latency") -> Graph:
    """Adapter: the research demo graph (ignores work_dir; keeps the measure_workflow signature)."""
    return _research_build(keyword=keyword)


_CLAUSES = (
    ("Term and Renewal", "renewal"),
    ("Fees and Payment", "payment"),
    ("Termination", "termination"),
    ("Confidentiality", "confidentiality"),
    ("Limitation of Liability", "liability"),
    ("Indemnification", "indemnity"),
    ("Governing Law", "jurisdiction"),
    ("Data Protection", "data"),
)


def _write_contract(work_dir: Path, n_repeats: int = 40) -> Path:
    """Write a deterministic multi-section contract (large intermediate for the comparison).

    Each clause is repeated across ``n_repeats`` numbered exhibits so the parsed ``Document`` is
    sizeable (kept in the cache as a handle), while the ``outline`` report stays tiny.
    """
    path = work_dir / "contract.txt"
    if not path.exists():
        lines = ["# Master Services Agreement", ""]
        for exhibit in range(n_repeats):
            for i, (heading, term) in enumerate(_CLAUSES, start=1):
                lines.append(f"## Exhibit {exhibit}.{i} {heading}")
                lines.append(
                    f"This {heading.lower()} clause (exhibit {exhibit}) governs {term}. "
                    f"The parties agree to standard {term} terms as set out in this section, "
                    f"which the reviewer must assess for risk and deviation from the playbook."
                )
                lines.append("")
        path.write_text("\n".join(lines) + "\n")
    return path


def build_docreview_graph(work_dir: Path, *, contains: str = "indemnity") -> Graph:
    """The document-review workflow on the real kit — read_document → filter_sections → outline.

    Deterministic (no LLM) so it runs headless in this harness; it still exercises a large
    ``Document`` artifact flowing through the graph, which is the point of the comparison. The
    LLM-backed ops (extract_clauses / classify_risk / …) are covered by the live smoke test.
    """
    path = _write_contract(work_dir)
    g = Graph()
    g.add(Node(id="doc", op="read_document",
               params={"path": str(path), "content_fp": content_hash(path)}))
    g.add(Node(id="filtered", op="filter_sections",
               params={"contains": contains}, inputs={"doc": "doc"}))
    g.add(Node(id="report", op="outline",
               params={"title": f"Sections mentioning {contains!r}"}, inputs={"doc": "filtered"}))
    return g


def _transcript_context(graph: Graph, values: dict[str, Value], cache: CacheStore) -> str:
    """What a transcript agent accrues: every node's full result inlined, in dependency order."""
    blocks: list[str] = []
    for nid in render.topo_order(graph):
        payload = cache.get(values[nid].fingerprint)
        if payload is None:
            continue
        blocks.append(f"[tool: {graph.nodes[nid].op}] {nid} result:\n{payload.decode()}")
    return "\n\n".join(blocks)


def _graph_context(
    tool: GraphPatch, graph: Graph, values: dict[str, Value], cache: CacheStore, report: Report
) -> str:
    """What the graph sends the model — the real :meth:`GraphPatch.get_llm_content` output."""
    collected = GraphPatch._collect_outputs(graph, values, cache)
    outputs = {nid: text[:_MAX_OUTPUT_CHARS] for nid, (_, text) in collected.items()}
    handles = {
        nid: OutputHandle(
            type=value.type,
            fingerprint=value.fingerprint,
            preview=text.strip()[:_MAX_OUTPUT_CHARS],
        )
        for nid, (value, text) in collected.items()
    }
    result = GraphPatchResult(
        applied=True,
        fresh=report.fresh(),
        cached=report.cached(),
        outputs=outputs,
        handles=handles,
        catalog=_render_catalog(),
    )
    content = tool.get_llm_content(result)
    assert content is not None  # get_llm_content returns text for an applied result
    return content


@dataclass
class TurnMetric:
    transcript_tokens: int
    graph_tokens: int
    transcript_nodes: int
    graph_nodes: int

    @property
    def reduction_pct(self) -> float:
        if self.transcript_tokens == 0:
            return 0.0
        return 100.0 * (1 - self.graph_tokens / self.transcript_tokens)


@dataclass
class Metrics:
    name: str
    node_count: int
    edit_label: str
    cold: TurnMetric
    rerun: TurnMetric


@contextmanager
def _cache_dir(work_dir: Path | None) -> Iterator[Path]:
    """Yield a directory for the cache — a caller-supplied one, or a fresh temp dir."""
    if work_dir is not None:
        yield work_dir
    else:
        with tempfile.TemporaryDirectory(prefix="metier-compare-") as tmp:
            yield Path(tmp)


async def measure_workflow(
    name: str,
    build_fn: Callable[..., Graph],
    edit_kwargs: dict,
    edit_label: str,
    *,
    work_dir: Path | None = None,
) -> Metrics:
    """Run ``build_fn`` cold then after ``edit_kwargs`` on one shared cache; measure both strategies.

    ``build_fn(**overrides)`` returns the graph; the same ``GraphPatch`` instance is reused
    across both turns so its "catalog once" behavior matches a real session (catalog on the
    cold turn, a one-line pointer on the re-run). ``work_dir`` overrides where the cache lives
    (tests pass a ``tmp_path``); it defaults to a fresh temp dir.
    """
    tool = _tool()
    with _cache_dir(work_dir) as tmp:
        cache = CacheStore(tmp / "cache.sqlite")
        try:
            # Cold run — both strategies compute every node.
            graph = build_fn(tmp)
            values, report = await execute(graph, cache)
            cold = TurnMetric(
                transcript_tokens=approx_token_count(_transcript_context(graph, values, cache)),
                graph_tokens=approx_token_count(_graph_context(tool, graph, values, cache, report)),
                transcript_nodes=len(graph.nodes),  # transcript has no cache: all nodes run
                graph_nodes=len(report.fresh()),
            )

            # Re-run after editing one param — only the dirty subgraph recomputes for the graph.
            graph2 = build_fn(tmp, **edit_kwargs)
            values2, report2 = await execute(graph2, cache)
            rerun = TurnMetric(
                transcript_tokens=approx_token_count(_transcript_context(graph2, values2, cache)),
                graph_tokens=approx_token_count(_graph_context(tool, graph2, values2, cache, report2)),
                transcript_nodes=len(graph2.nodes),  # transcript recomputes + re-inlines everything
                graph_nodes=len(report2.fresh()),
            )
        finally:
            cache.close()
    return Metrics(name=name, node_count=len(graph.nodes), edit_label=edit_label, cold=cold, rerun=rerun)


def _print_metrics(m: Metrics) -> None:
    print(f"\n=== {m.name} workflow ({m.node_count} nodes) ===")
    print(f"{'':<26}{'transcript':>14}{'graph':>14}{'reduction':>12}")
    for label, turn in (("cold-run context", m.cold), ("re-run context", m.rerun)):
        print(
            f"{label:<26}{turn.transcript_tokens:>11,} tok{turn.graph_tokens:>11,} tok"
            f"{turn.reduction_pct:>11.1f}%"
        )
    total_t = m.cold.transcript_tokens + m.rerun.transcript_tokens
    total_g = m.cold.graph_tokens + m.rerun.graph_tokens
    reduction = 100.0 * (1 - total_g / total_t) if total_t else 0.0
    print(f"{'cumulative context':<26}{total_t:>11,} tok{total_g:>11,} tok{reduction:>11.1f}%")
    print(
        f"nodes computed on re-run   {m.rerun.transcript_nodes:>11}   {m.rerun.graph_nodes:>12}"
        f"   (edited: {m.edit_label})"
    )


async def main() -> None:
    print(
        "Métier graph vs. a transcript agent — same computation, two strategies.\n"
        "Transcript = every tool result inlined, no cache (models the 'original version').\n"
        "Graph = handles in context, payloads in cache, incremental re-run.\n"
        "Token counts are approximate (chars/4)."
    )
    analytics = await measure_workflow(
        "analytics",
        build_analytics_graph,
        edit_kwargs={"key": "region"},
        edit_label="group_by key country → region",
    )
    research = await measure_workflow(
        "research",
        build_research_graph,
        edit_kwargs={"keyword": "pricing"},
        edit_label="extract keyword latency → pricing",
    )
    doc_review = await measure_workflow(
        "doc-review",
        build_docreview_graph,
        edit_kwargs={"contains": "termination"},
        edit_label="filter sections indemnity → termination",
    )
    for metrics in (analytics, research, doc_review):
        _print_metrics(metrics)


if __name__ == "__main__":
    asyncio.run(main())
