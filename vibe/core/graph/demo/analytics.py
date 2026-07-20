"""A deterministic analytics operator kit + pipeline for the comparison demo.

This is the "real use case": a small data-analytics DAG whose *intermediate* tables are
large (thousands of rows). It mirrors how a dbt / spreadsheet pipeline is shaped —
load → filter → join → group → rank → report — and every operator is a pure function of
its declared inputs and params, so the content-addressed cache is sound and re-runs
recompute only the dirty subgraph.

The data is generated deterministically from a seed (:func:`_rng`), so node fingerprints
— and therefore cache hits and the harness's measured numbers — are stable across runs.
No API keys, no network: it runs headless in CI.

The DAG (:func:`build_analytics_graph`)::

    load_events(seed) ─ filter_events(action) ─┐
                                               ├─ join_users ─ group_by(key) ─ top_n(n) ─ analytics_report
    load_users(seed) ──────────────────────────┘
"""

from __future__ import annotations

import random
from typing import Any

from pydantic import BaseModel, Field

from vibe.core.graph.model import Graph, Node
from vibe.core.graph.operators import operator

_REGIONS = ("emea", "amer", "apac", "latam")
_ACTIONS = ("view", "signup", "purchase", "refund")
_PLANS = ("free", "pro", "enterprise")
_COUNTRIES = ("fr", "us", "de", "jp", "br", "in", "gb", "ca")


class EventTable(BaseModel):
    """A wide fact table — the kind of intermediate that is expensive to inline in context."""

    rows: list[dict[str, Any]] = Field(default_factory=list)


class UserTable(BaseModel):
    rows: list[dict[str, Any]] = Field(default_factory=list)


class GroupedTable(BaseModel):
    """Aggregated rows: ``key``, ``total`` (sum of amount), ``events`` (count)."""

    rows: list[dict[str, Any]] = Field(default_factory=list)


class AnalyticsReport(BaseModel):
    markdown: str


def _rng(seed: int) -> random.Random:
    """A seeded RNG — deterministic output for a given seed keeps fingerprints stable."""
    return random.Random(seed)


@operator
async def load_events(seed: int, n_rows: int) -> EventTable:
    """Generate a synthetic event fact table (deterministic in ``seed``)."""
    rng = _rng(seed)
    rows = [
        {
            "event_id": i,
            "user_id": rng.randint(0, max(0, n_rows // 4)),
            "region": rng.choice(_REGIONS),
            "action": rng.choice(_ACTIONS),
            "amount": round(rng.uniform(1.0, 500.0), 2),
        }
        for i in range(n_rows)
    ]
    return EventTable(rows=rows)


@operator
async def load_users(seed: int, n_users: int) -> UserTable:
    """Generate a synthetic user dimension table (deterministic in ``seed``)."""
    rng = _rng(seed)
    rows = [
        {
            "user_id": i,
            "plan": rng.choice(_PLANS),
            "country": rng.choice(_COUNTRIES),
        }
        for i in range(n_users)
    ]
    return UserTable(rows=rows)


@operator
async def filter_events(events: EventTable, action: str) -> EventTable:
    """Keep only rows whose ``action`` matches (a WHERE clause over the fact table)."""
    return EventTable(rows=[r for r in events.rows if r["action"] == action])


@operator
async def join_users(events: EventTable, users: UserTable) -> EventTable:
    """Left-join user attributes (plan, country) onto each event by ``user_id``."""
    by_id = {u["user_id"]: u for u in users.rows}
    joined: list[dict[str, Any]] = []
    for row in events.rows:
        user = by_id.get(row["user_id"], {})
        joined.append({**row, "plan": user.get("plan", "unknown"), "country": user.get("country", "??")})
    return EventTable(rows=joined)


@operator
async def group_by(events: EventTable, key: str) -> GroupedTable:
    """Aggregate sum(amount) and event count grouped by ``key`` (e.g. country, plan, region)."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in events.rows:
        k = str(row.get(key, "??"))
        totals[k] = totals.get(k, 0.0) + float(row.get("amount", 0.0))
        counts[k] = counts.get(k, 0) + 1
    rows = [
        {"key": k, "total": round(totals[k], 2), "events": counts[k]}
        for k in sorted(totals, key=lambda k: totals[k], reverse=True)
    ]
    return GroupedTable(rows=rows)


@operator
async def top_n(grouped: GroupedTable, n: int) -> GroupedTable:
    """Keep the top ``n`` groups by total (a LIMIT after ORDER BY total DESC)."""
    return GroupedTable(rows=grouped.rows[:n])


@operator
async def analytics_report(grouped: GroupedTable, title: str) -> AnalyticsReport:
    """Render the ranked groups as a compact markdown brief."""
    lines = [f"# {title}", ""]
    lines.extend(
        f"- {row['key']}: {row['total']:,.2f} across {row['events']} events" for row in grouped.rows
    )
    return AnalyticsReport(markdown="\n".join(lines))


def build_analytics_graph(
    *,
    seed: int = 7,
    n_rows: int = 2000,
    n_users: int = 500,
    action: str = "purchase",
    key: str = "country",
    n: int = 5,
    title: str = "Top markets by purchase revenue",
) -> Graph:
    """Wire the load → filter → join → group → rank → report analytics DAG."""
    graph = Graph()
    graph.add(Node(id="events", op="load_events", params={"seed": seed, "n_rows": n_rows}))
    graph.add(Node(id="users", op="load_users", params={"seed": seed, "n_users": n_users}))
    graph.add(Node(id="filtered", op="filter_events", params={"action": action}, inputs={"events": "events"}))
    graph.add(
        Node(id="enriched", op="join_users", inputs={"events": "filtered", "users": "users"})
    )
    graph.add(Node(id="grouped", op="group_by", params={"key": key}, inputs={"events": "enriched"}))
    graph.add(Node(id="ranked", op="top_n", params={"n": n}, inputs={"grouped": "grouped"}))
    graph.add(
        Node(id="report", op="analytics_report", params={"title": title}, inputs={"grouped": "ranked"})
    )
    return graph
