"""The data-analysis operator library — the ``analyst`` agent's toolkit.

One generic, composable :class:`Table` (`columns` + `rows`) flows through every operator, so
they chain in any order: load → clean → derive → join → aggregate → analyze → report. All
operators are tagged ``library="analysis"`` for catalog scoping, and every error names the
offending column and lists the available ones — the agent recovers by reading the feedback.

Typing: `read_csv`/`sample_dataset` infer a column numeric iff every non-empty cell parses
(int, else float); `cast_column` overrides; aggregations raise on a non-numeric metric rather
than coercing silently.
"""

from __future__ import annotations

import csv
from datetime import date, datetime
import importlib.resources
from io import StringIO
from pathlib import Path
import statistics
from typing import Any

from pydantic import BaseModel, Field

from vibe.core.graph.blocks import BlockDef, is_block, register_block
from vibe.core.graph.model import Graph, Node
from vibe.core.graph.operators import operator

_LIB = "analysis"
_MAX_CSV_BYTES = 50 * 1024 * 1024  # refuse files bigger than this (guard against OOM)
_MAX_CSV_ROWS = 500_000
_SAMPLE_DATASETS = ("sales", "customers")
_WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_NUMERIC_AGGS = ("sum", "mean", "min", "max")  # + "count" handled separately


class Table(BaseModel):
    """A tabular value: an ordered column list and rows keyed by exactly those columns."""

    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)


class Report(BaseModel):
    markdown: str


# --- invariant + error helpers -------------------------------------------------------------


def _table(columns: list[str], rows: list[dict[str, Any]]) -> Table:
    """Normalize to the invariant: every row keyed by *exactly* ``columns`` (missing → None)."""
    cols = list(columns)
    return Table(columns=cols, rows=[{c: r.get(c) for c in cols} for r in rows])


def _cols(value: list[str] | str) -> list[str]:
    """Accept a single column name or a list — a bare string is one column, not per-character."""
    return [value] if isinstance(value, str) else list(value)


def _need(table: Table, *columns: str) -> None:
    """Raise an actionable error if any named column is absent."""
    for col in columns:
        if col not in table.columns:
            raise ValueError(
                f"column {col!r} not found; available columns are {table.columns}"
            )


def _column_is_numeric(table: Table, column: str) -> bool:
    vals = [r[column] for r in table.rows if r[column] is not None]
    return bool(vals) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals)


def _require_numeric(table: Table, column: str, op: str) -> None:
    _need(table, column)
    if not _column_is_numeric(table, column):
        raise ValueError(
            f"{op}: column {column!r} is not numeric; cast_column it to int/float first"
        )


# --- CSV parsing + type inference ----------------------------------------------------------


def _looks_zero_padded(v: str) -> bool:
    """A zero-padded identifier (zip, id) — must stay text; parsing it to int loses the zero."""
    return len(v) > 1 and v[0] == "0" and v[1] != "."


def _coerce_column(values: list[str]) -> list[Any]:
    """Infer int/float/str for a column of raw strings; '' → None."""
    non_empty = [v for v in values if v != ""]

    def _all(parse: Any) -> bool:
        try:
            for v in non_empty:
                parse(v)
        except ValueError:
            return False
        return bool(non_empty)

    # Zero-padded values (zip codes, ids) look numeric but must stay text — coercing '01234' to
    # 1234 silently corrupts the identifier, and cast_column can't recover the lost zero.
    if any(_looks_zero_padded(v) for v in non_empty):
        return [v if v != "" else None for v in values]
    if _all(int):
        return [int(v) if v != "" else None for v in values]
    if _all(float):
        return [float(v) if v != "" else None for v in values]
    return [v if v != "" else None for v in values]


def _parse_csv(text: str) -> Table:
    reader = csv.DictReader(StringIO(text))
    columns = list(reader.fieldnames or [])
    raw = [dict(r) for r in reader]
    if len(raw) > _MAX_CSV_ROWS:
        raise ValueError(f"CSV has {len(raw)} rows; over the {_MAX_CSV_ROWS}-row cap — sample it first")
    typed_cols = {c: _coerce_column([str(r.get(c, "") or "") for r in raw]) for c in columns}
    rows = [{c: typed_cols[c][i] for c in columns} for i in range(len(raw))]
    return _table(columns, rows)


# --- load ----------------------------------------------------------------------------------


@operator(library=_LIB, reads_file="path")
async def read_csv(path: str, content_fp: str) -> Table:
    """Load a CSV file into a table (column types inferred). Supply only ``path`` — the tool
    fingerprints the file (``content_fp``) so an edit re-runs the dependent subgraph.
    """
    p = Path(path).expanduser()
    try:
        size = p.stat().st_size
    except OSError as exc:
        raise ValueError(f"read_csv: cannot read {path!r}: {exc}") from exc
    if size > _MAX_CSV_BYTES:
        raise ValueError(
            f"read_csv: {path!r} is {size} bytes, over the {_MAX_CSV_BYTES}-byte cap — sample it first"
        )
    return _parse_csv(p.read_text())


@operator(library=_LIB)
async def sample_dataset(name: str) -> Table:
    """Load a bundled sample dataset by name (no file path needed). Datasets: sales, customers."""
    if name not in _SAMPLE_DATASETS:
        raise ValueError(f"unknown sample dataset {name!r}; available: {list(_SAMPLE_DATASETS)}")
    text = (importlib.resources.files("vibe.core.graph.library.data") / f"{name}.csv").read_text()
    return _parse_csv(text)


# --- clean / shape -------------------------------------------------------------------------


@operator(library=_LIB)
async def select_columns(table: Table, columns: list[str]) -> Table:
    """Keep only the named columns, in the given order."""
    columns = _cols(columns)
    _need(table, *columns)
    return _table(columns, table.rows)


@operator(library=_LIB)
async def rename_columns(table: Table, mapping: dict[str, str]) -> Table:
    """Rename columns via an {old: new} mapping."""
    _need(table, *mapping.keys())
    new_cols = [mapping.get(c, c) for c in table.columns]
    rows = [{mapping.get(c, c): r[c] for c in table.columns} for r in table.rows]
    return _table(new_cols, rows)


@operator(library=_LIB)
async def cast_column(table: Table, column: str, type: str) -> Table:
    """Cast a column to ``int``, ``float``, or ``str`` (blank/invalid → None for numerics)."""
    _need(table, column)
    if type not in {"int", "float", "str"}:
        raise ValueError(f"cast_column: type must be int/float/str, got {type!r}")

    def cast(v: Any) -> Any:
        if v is None or v == "":
            return None if type != "str" else ""
        try:
            return {"int": lambda x: int(float(x)), "float": float, "str": str}[type](v)
        except (ValueError, TypeError):
            if type == "str":
                return str(v)
            return None

    rows = [{**r, column: cast(r[column])} for r in table.rows]
    return _table(table.columns, rows)


@operator(library=_LIB)
async def drop_missing(table: Table, columns: list[str]) -> Table:
    """Drop rows with a missing (None/blank) value in any of ``columns`` (empty → all columns)."""
    check = _cols(columns) or table.columns
    _need(table, *check)
    kept = [r for r in table.rows if all(r[c] is not None and r[c] != "" for c in check)]
    return _table(table.columns, kept)


def _compare(cell: Any, op: str, value: Any) -> bool:
    if op == "contains":
        return value.lower() in str(cell).lower() if cell is not None else False
    if cell is None:
        return False
    ops = {
        "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
        ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
        "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
    }
    return ops[op](cell, value)


@operator(library=_LIB)
async def filter_rows(table: Table, column: str, op: str, value: str) -> Table:
    """Keep rows where ``column`` ``op`` ``value``. op ∈ ==, !=, >, >=, <, <=, contains."""
    _need(table, column)
    valid = ("==", "!=", ">", ">=", "<", "<=", "contains")
    if op not in valid:
        raise ValueError(f"filter_rows: op must be one of {list(valid)}, got {op!r}")
    typed: Any = value
    if op != "contains" and _column_is_numeric(table, column):
        try:
            typed = float(value)
        except ValueError as exc:
            raise ValueError(f"filter_rows: {column!r} is numeric but value {value!r} isn't") from exc
    kept = [r for r in table.rows if _compare(r[column], op, typed)]
    return _table(table.columns, kept)


def _sorted_non_null_first(rows: list[dict[str, Any]], by: str, descending: bool) -> list[dict[str, Any]]:
    """Sort by ``by`` with ``None`` always last (in both directions), so nulls never rank as top."""
    present = [r for r in rows if r[by] is not None]
    missing = [r for r in rows if r[by] is None]
    present.sort(key=lambda r: r[by], reverse=descending)
    return present + missing


@operator(library=_LIB)
async def sort_rows(table: Table, by: str, descending: bool) -> Table:
    """Sort rows by a column (None always sorts last)."""
    _need(table, by)
    return _table(table.columns, _sorted_non_null_first(table.rows, by, descending))


@operator(library=_LIB)
async def limit(table: Table, n: int) -> Table:
    """Keep the first ``n`` rows."""
    return _table(table.columns, table.rows[: max(0, n)])


@operator(library=_LIB)
async def distinct(table: Table, columns: list[str]) -> Table:
    """Drop duplicate rows (by ``columns``, or all columns when empty)."""
    keys = _cols(columns) or table.columns
    _need(table, *keys)
    seen: set[tuple[Any, ...]] = set()
    kept: list[dict[str, Any]] = []
    for r in table.rows:
        sig = tuple(r[c] for c in keys)
        if sig not in seen:
            seen.add(sig)
            kept.append(r)
    return _table(table.columns, kept)


@operator(library=_LIB)
async def derive_column(table: Table, name: str, left: str, op: str, right: str) -> Table:
    """Add ``name`` = ``left`` <op> ``right`` (op ∈ +,-,*,/). ``right`` is a numeric constant or
    another column name.
    """
    _require_numeric(table, left, "derive_column")
    if op not in {"+", "-", "*", "/"}:
        raise ValueError(f"derive_column: op must be +,-,*,/, got {op!r}")
    const: float | None = None
    try:
        const = float(right)
    except ValueError:
        _require_numeric(table, right, "derive_column")

    def compute(r: dict[str, Any]) -> Any:
        a = r[left]
        b = const if const is not None else r[right]
        if a is None or b is None:
            return None
        if op == "/":
            return a / b if b != 0 else None
        return {"+": a + b, "-": a - b, "*": a * b}[op]

    cols = table.columns + ([name] if name not in table.columns else [])
    rows = [{**r, name: compute(r)} for r in table.rows]
    return _table(cols, rows)


@operator(library=_LIB)
async def date_part(table: Table, column: str, part: str) -> Table:
    """Add a column ``{column}_{part}`` extracted from an ISO date. part ∈ year, month, day,
    weekday.
    """
    _need(table, column)
    if part not in {"year", "month", "day", "weekday"}:
        raise ValueError(f"date_part: part must be year/month/day/weekday, got {part!r}")

    def extract(v: Any) -> Any:
        if v is None or v == "":
            return None
        try:
            d = datetime.fromisoformat(str(v)).date() if "T" in str(v) else date.fromisoformat(str(v))
        except ValueError as exc:
            raise ValueError(f"date_part: {v!r} in {column!r} is not an ISO date") from exc
        if part == "weekday":
            return _WEEKDAYS[d.weekday()]
        return getattr(d, part)

    new = f"{column}_{part}"
    cols = table.columns + ([new] if new not in table.columns else [])
    rows = [{**r, new: extract(r[column])} for r in table.rows]
    return _table(cols, rows)


# --- combine -------------------------------------------------------------------------------


@operator(library=_LIB)
async def join(left: Table, right: Table, on: str, how: str) -> Table:
    """Join two tables on a shared column. how ∈ inner, left."""
    _need(left, on)
    _need(right, on)
    if how not in {"inner", "left"}:
        raise ValueError(f"join: how must be inner/left, got {how!r}")
    right_cols = [c for c in right.columns if c != on]
    # This is a lookup join (one right row per key), not a many-to-many join. Duplicate keys on
    # the right would silently drop matches, so reject them with an actionable error.
    right_keys = [r[on] for r in right.rows]
    if len(set(right_keys)) != len(right_keys):
        raise ValueError(
            f"join: right table has duplicate {on!r} values; deduplicate it first "
            "(e.g. distinct or group_by) so each key maps to one row"
        )
    index: dict[Any, dict[str, Any]] = {r[on]: r for r in right.rows}
    out_cols = left.columns + [c for c in right_cols if c not in left.columns]
    rows: list[dict[str, Any]] = []
    for lr in left.rows:
        match = index.get(lr[on])
        if match is None:
            if how == "inner":
                continue
            rows.append({**lr, **{c: None for c in right_cols}})
        else:
            rows.append({**lr, **{c: match[c] for c in right_cols}})
    return _table(out_cols, rows)


# --- aggregate / analyze -------------------------------------------------------------------


def _agg(values: list[float], how: str) -> float:
    if how == "sum":
        return sum(values)
    if how == "mean":
        return statistics.fmean(values) if values else 0.0
    if how == "min":
        return min(values)
    if how == "max":
        return max(values)
    raise ValueError(f"unknown aggregation {how!r}")


@operator(library=_LIB)
async def group_by(table: Table, keys: list[str], metric: str, aggs: list[str]) -> Table:
    """Group by ``keys`` (empty → overall total) and aggregate ``metric``. aggs ⊆ sum, mean, min,
    max, count. Output columns: keys + one per agg (``count`` is a row count).
    """
    keys = _cols(keys)
    _need(table, *keys)
    unknown = [a for a in aggs if a not in {*_NUMERIC_AGGS, "count"}]
    if unknown:
        raise ValueError(f"group_by: unknown aggs {unknown}; use sum/mean/min/max/count")
    numeric = [a for a in aggs if a != "count"]
    if numeric:
        _require_numeric(table, metric, "group_by")

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    order: list[tuple[Any, ...]] = []
    for r in table.rows:
        sig = tuple(r[k] for k in keys)
        if sig not in groups:
            groups[sig] = []
            order.append(sig)
        groups[sig].append(r)

    out_cols = [*keys, *(("count",) if "count" in aggs else ()), *(f"{metric}_{a}" for a in numeric)]
    rows: list[dict[str, Any]] = []
    for sig in order:
        members = groups[sig]
        row = dict(zip(keys, sig, strict=True))
        if "count" in aggs:
            row["count"] = len(members)
        vals = [m[metric] for m in members if m[metric] is not None]
        for a in numeric:
            row[f"{metric}_{a}"] = round(_agg([float(v) for v in vals], a), 4) if vals else None
        rows.append(row)
    return _table(out_cols, rows)


@operator(library=_LIB)
async def describe(table: Table, columns: list[str]) -> Table:
    """Summary stats (count, mean, std, min, max) per numeric column (empty → all numeric)."""
    cols = _cols(columns) or [c for c in table.columns if _column_is_numeric(table, c)]
    _need(table, *cols)
    out_cols = ["column", "count", "mean", "std", "min", "max"]
    rows: list[dict[str, Any]] = []
    for c in cols:
        _require_numeric(table, c, "describe")
        vals = [float(r[c]) for r in table.rows if r[c] is not None]
        rows.append({
            "column": c,
            "count": len(vals),
            "mean": round(statistics.fmean(vals), 4) if vals else None,
            "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
        })
    return _table(out_cols, rows)


@operator(library=_LIB)
async def value_counts(table: Table, column: str) -> Table:
    """Frequency of each distinct value in ``column``, most frequent first."""
    _need(table, column)
    counts: dict[Any, int] = {}
    for r in table.rows:
        counts[r[column]] = counts.get(r[column], 0) + 1
    rows = [{"value": k, "count": v} for k, v in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)]
    return _table(["value", "count"], rows)


@operator(library=_LIB)
async def top_n(table: Table, by: str, n: int) -> Table:
    """The top ``n`` rows by ``by`` (descending; None-valued rows never count as top)."""
    _need(table, by)
    ranked = _sorted_non_null_first(table.rows, by, descending=True)
    return _table(table.columns, ranked[: max(0, n)])


# --- report --------------------------------------------------------------------------------


@operator(library=_LIB)
async def to_markdown(table: Table, title: str, max_rows: int) -> Report:
    """Render the table as a markdown report (first ``max_rows`` rows)."""

    def cell(v: Any) -> str:
        return "" if v is None else str(v).replace("|", "\\|")

    lines = [f"# {title}", ""]
    if not table.columns:
        lines.append("_(empty)_")
        return Report(markdown="\n".join(lines))
    lines.append("| " + " | ".join(table.columns) + " |")
    lines.append("| " + " | ".join("---" for _ in table.columns) + " |")
    for r in table.rows[: max(0, max_rows)]:
        lines.append("| " + " | ".join(cell(r[c]) for c in table.columns) + " |")
    if len(table.rows) > max_rows:
        lines.append("")
        lines.append(f"_… {len(table.rows) - max_rows} more rows_")
    return Report(markdown="\n".join(lines))


# --- classic-analysis blocks (the workflows, encoded) --------------------------------------


def _quick_profile() -> Graph:
    g = Graph()
    g.add(Node(id="d", op="describe", params={"columns": []}))
    g.add(Node(id="r", op="to_markdown", params={"max_rows": 50}, inputs={"table": "d"}))
    return g


def _rank_by() -> Graph:
    g = Graph()
    g.add(Node(id="g", op="group_by", params={"keys": [], "metric": "", "aggs": ["sum"]}))
    g.add(Node(id="t", op="top_n", params={"by": "", "n": 5}, inputs={"table": "g"}))
    g.add(Node(id="r", op="to_markdown", params={"max_rows": 50}, inputs={"table": "t"}))
    return g


def _trend_by_period() -> Graph:
    g = Graph()
    g.add(Node(id="dp", op="date_part", params={"column": "", "part": "month"}))
    g.add(Node(id="g", op="group_by", params={"keys": [], "metric": "", "aggs": ["sum"]}, inputs={"table": "dp"}))
    g.add(Node(id="s", op="sort_rows", params={"by": "", "descending": False}, inputs={"table": "g"}))
    g.add(Node(id="r", op="to_markdown", params={"max_rows": 100}, inputs={"table": "s"}))
    return g


def _segment_summary() -> Graph:
    g = Graph()
    g.add(Node(id="g", op="group_by", params={"keys": [], "metric": "", "aggs": ["count", "sum", "mean"]}))
    g.add(Node(id="r", op="to_markdown", params={"max_rows": 100}, inputs={"table": "g"}))
    return g


_BLOCKS = [
    BlockDef(
        name="quick_profile", graph=_quick_profile(), library=_LIB,
        description="summary stats for every numeric column",
        input_ports={"table": ("d", "table")}, params={"title": ("r", "title")}, output="r",
    ),
    BlockDef(
        name="rank_by", graph=_rank_by(), library=_LIB,
        description="top N groups by a summed metric; rank_by_column is <metric>_sum",
        input_ports={"table": ("g", "table")},
        params={"group_key": ("g", "keys"), "metric": ("g", "metric"),
                "rank_by_column": ("t", "by"), "n": ("t", "n"), "title": ("r", "title")},
        output="r",
    ),
    BlockDef(
        name="trend_by_period", graph=_trend_by_period(), library=_LIB,
        description="metric summed per calendar period (time series)",
        input_ports={"table": ("dp", "table")},
        params={"date_column": ("dp", "column"), "period": ("dp", "part"),
                "group_key": ("g", "keys"), "metric": ("g", "metric"),
                "sort_key": ("s", "by"), "title": ("r", "title")},
        output="r",
    ),
    BlockDef(
        name="segment_summary", graph=_segment_summary(), library=_LIB,
        description="count/sum/mean of a metric per segment",
        input_ports={"table": ("g", "table")},
        params={"segment": ("g", "keys"), "metric": ("g", "metric"), "title": ("r", "title")},
        output="r",
    ),
]


def register() -> None:
    """Register the analysis blocks (idempotent). Operators register on import above."""
    for block in _BLOCKS:
        if not is_block(block.name):
            register_block(block)


register()
