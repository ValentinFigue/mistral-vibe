"""Runnable demo: ``python -m vibe.core.graph.demo``.

Runs the weekly-margin pipeline three times against one on-disk cache and prints a
per-node table each time:

* run 1 — cold cache, every node ``fresh``;
* run 2 — warm cache, every node ``cached`` in near-zero time;
* run 3 — after editing ``sales.csv``, only the dirty subgraph is ``fresh``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile

from vibe.core.graph.cache import CacheStore
from vibe.core.graph.demo.pipeline import build_graph, write_fixtures
from vibe.core.graph.executor import execute
from vibe.core.graph.model import Graph, Report


def _print_report(label: str, report: Report) -> None:
    print(f"\n=== {label} ===")
    print(f"{'node':<14}{'state':<9}{'ms':>8}")
    for node_id, state in report.states.items():
        ms = report.timings[node_id] * 1000
        marker = "  " if state == "cached" else "* "
        print(f"{marker}{node_id:<12}{state:<9}{ms:>8.2f}")
    print(f"fresh={report.fresh()}  cached={report.cached()}")


async def _run(graph: Graph, cache: CacheStore, label: str) -> None:
    _, report = await execute(graph, cache)
    _print_report(label, report)


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="metier-demo-") as tmp:
        work_dir = Path(tmp)
        sales, costs = write_fixtures(work_dir)
        cache = CacheStore(work_dir / "cache.sqlite")

        graph = build_graph(sales, costs)
        await _run(graph, cache, "run 1 — cold cache (all fresh)")
        await _run(graph, cache, "run 2 — warm cache (all cached)")

        # Edit one input, rebuild the graph (source fingerprints fold in file content).
        sales.write_text("region,revenue\nemea,1500\namer,2100\napac,900\n")
        graph = build_graph(sales, costs)
        await _run(graph, cache, "run 3 — edited sales.csv (dirty subgraph only)")

        cache.close()


if __name__ == "__main__":
    asyncio.run(main())
