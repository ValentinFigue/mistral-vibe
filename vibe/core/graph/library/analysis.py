"""The data-analysis operator library — the ``analyst`` agent's toolkit.

One generic, composable :class:`Table` (`columns` + `rows`) flows through every operator, so
they chain in any order: load → clean → derive → join → aggregate → analyze → report → export.
All operators are tagged ``library="analysis"`` for catalog scoping, and every error names the
offending column and lists the available ones — the agent recovers by reading the feedback.

**Engine.** Aggregation, reshape, stats, window and join operators run on **pandas** (numpy
under it), via the `_to_df`/`_from_df` bridge; the elementwise/row operators (select, filter,
sort, cast, derive, date_part, …) stay pure-Python (already correct and cheap, and avoids
NaN/dtype drift). pandas/matplotlib are imported **lazily inside op bodies**, never at module
top, so importing this module for tool discovery / catalog rendering (every CLI startup) does
not load them. The `{columns, rows}` wire format is unchanged: `_from_df` round-trips through
JSON so cached values stay JSON-native (no numpy scalars, NaN→null, ISO dates), keeping the
content-addressed cache, `graph_inspect`, and `verify_purity` equality all sound.

Typing: `read_csv`/`sample_dataset` infer a column numeric iff every non-empty cell parses
(int, else float) and keep zero-padded ids as strings; `cast_column` overrides; aggregations
raise on a non-numeric metric rather than coercing silently. `join` is a real merge (duplicate
keys multiply). Sink operators (`to_csv`, `bar_chart`, `line_chart`) write a file and return a
small handle, keeping bytes out of context.
"""

from __future__ import annotations

from collections.abc import Sequence
import csv
from datetime import date, datetime
import importlib.resources
from io import StringIO
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from vibe.core.graph.blocks import BlockDef, is_block, register_block
from vibe.core.graph.model import Graph, Node
from vibe.core.graph.operators import operator
from vibe.core.llm.types import (
    LLMCaller,  # lightweight (protocol); the `llm` param is executor-injected
)

_LIB = "analysis"
_MAX_CSV_BYTES = 50 * 1024 * 1024  # refuse files bigger than this (guard against OOM)
_MAX_CSV_ROWS = 500_000
_SAMPLE_DATASETS = ("sales", "customers")
_WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_NUMERIC_AGGS = ("sum", "mean", "min", "max")  # + "count" handled separately

# Catalog sub-groups (rendered as [category] headers, ordered by render._CATEGORY_ORDER) — one place
# to keep the taxonomy that the analyst prompt mirrors. Every analysis op is tagged here via `_op`.
_CATEGORY_BY_OP = {
    # load
    "read_csv": "load", "sample_dataset": "load",
    # clean / reshape / derive
    "select_columns": "shape", "rename_columns": "shape", "drop_missing": "shape",
    "filter_rows": "shape", "sort_rows": "shape", "limit": "shape", "distinct": "shape",
    "derive_column": "shape", "date_part": "shape", "join": "shape", "concat": "shape",
    "pivot": "shape", "melt": "shape", "group_by": "shape", "top_n": "shape", "rank": "shape",
    "bin": "shape", "pct_change": "shape", "rolling": "shape",
    # feature transforms
    "cast_column": "transform", "fill_missing": "transform", "normalize": "transform",
    "encode": "transform",
    # descriptive statistics
    "describe": "statistics", "quantile": "statistics", "distribution": "statistics",
    "correlation": "statistics", "value_counts": "statistics", "outliers": "statistics",
    # inferential statistics
    "corr_test": "inference", "normality_test": "inference", "group_test": "inference",
    "chi_square": "inference",
    # machine learning
    "ml_regression": "ml", "ml_classification": "ml", "ml_cluster": "ml",
    "feature_importance": "ml", "ml_predict": "ml",
    # data-quality gates
    "expect_columns": "quality", "expect_no_nulls": "quality", "expect_unique": "quality",
    # sql
    "sql": "sql",
    # report / sink
    "to_markdown": "report", "to_csv": "report", "answer": "report", "chart": "report",
    # llm insight
    "narrate": "insight", "classify": "insight",
}


def _op(  # noqa: ANN202 (decorator factory)
    *, name: str | None = None, needs_llm: bool = False, reads_file: str | None = None
):
    """``@operator`` scoped to this library, auto-tagging the op's ``category`` from
    ``_CATEGORY_BY_OP`` so the catalog groups it — the single place the taxonomy lives.
    """
    def wrap(fn: Any) -> Any:
        op_name = name or fn.__name__
        return operator(
            library=_LIB, category=_CATEGORY_BY_OP.get(op_name),
            name=name, needs_llm=needs_llm, reads_file=reads_file,
        )(fn)

    return wrap


class Table(BaseModel):
    """A tabular value: an ordered column list and rows keyed by exactly those columns."""

    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)


class Report(BaseModel):
    markdown: str


class ExportResult(BaseModel):
    """A handle to a written CSV — the file is on disk; the value stays tiny (out of context)."""

    path: str
    rows: int
    columns: list[str]


class ChartResult(BaseModel):
    """A handle to a written chart image — the PNG is on disk, never inlined into context."""

    path: str
    kind: str


# --- invariant + error helpers -------------------------------------------------------------


def _table(columns: list[str], rows: list[dict[str, Any]]) -> Table:
    """Normalize to the invariant: every row keyed by *exactly* ``columns`` (missing → None)."""
    cols = list(columns)
    return Table(columns=cols, rows=[{c: r.get(c) for c in cols} for r in rows])


def _cols(value: Sequence[str] | str | None) -> list[str]:
    """Accept a single column name, a sequence, or None — a bare string is one column, not chars.

    Takes a covariant ``Sequence`` so enum lists (``list[Literal[...]]``) are accepted too.
    """
    if value is None:
        return []
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


# --- pandas bridge -------------------------------------------------------------------------
# pandas/numpy back the aggregation, reshape, stats and window operators. They are imported
# *lazily* here (never at module top) so that importing this module for tool discovery / catalog
# rendering — which happens on every vibe startup — does not load pandas. The `{columns, rows}`
# wire format is unchanged: `_from_df` round-trips through JSON so values stay JSON-native
# (no numpy scalars, NaN→null, ISO dates), which keeps the cache, graph_inspect, and
# verify_purity equality all working.


def _to_df(table: Table):  # noqa: ANN202 (pandas.DataFrame, imported lazily)
    import pandas as pd

    return pd.DataFrame(table.rows, columns=table.columns)


def _from_df(df) -> Table:  # noqa: ANN001
    import json

    frame = df.copy()
    frame.columns = [str(c) for c in frame.columns]
    rows = json.loads(frame.to_json(orient="records", date_format="iso"))
    return Table(columns=list(frame.columns), rows=rows)


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


@_op(reads_file="path")
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


@_op()
async def sample_dataset(name: str) -> Table:
    """Load a bundled sample dataset by name (no file path needed). Datasets: sales, customers."""
    if name not in _SAMPLE_DATASETS:
        raise ValueError(f"unknown sample dataset {name!r}; available: {list(_SAMPLE_DATASETS)}")
    text = (importlib.resources.files("vibe.core.graph.library.data") / f"{name}.csv").read_text()
    return _parse_csv(text)


# --- clean / shape -------------------------------------------------------------------------


@_op()
async def select_columns(table: Table, columns: list[str]) -> Table:
    """Keep only the named columns, in the given order."""
    columns = _cols(columns)
    _need(table, *columns)
    return _table(columns, table.rows)


@_op()
async def rename_columns(table: Table, mapping: dict[str, str]) -> Table:
    """Rename columns via an {old: new} mapping."""
    _need(table, *mapping.keys())
    new_cols = [mapping.get(c, c) for c in table.columns]
    rows = [{mapping.get(c, c): r[c] for c in table.columns} for r in table.rows]
    return _table(new_cols, rows)


@_op()
async def cast_column(table: Table, column: str, type: Literal["int", "float", "str"]) -> Table:
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


@_op()
async def drop_missing(table: Table, columns: list[str] | None = None) -> Table:
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


@_op()
async def filter_rows(
    table: Table,
    column: str,
    value: str,
    op: Literal["==", "!=", ">", ">=", "<", "<=", "contains"] = "==",
) -> Table:
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


@_op()
async def sort_rows(table: Table, by: str, descending: bool = False) -> Table:
    """Sort rows by a column (None always sorts last)."""
    _need(table, by)
    return _table(table.columns, _sorted_non_null_first(table.rows, by, descending))


@_op()
async def limit(table: Table, n: int) -> Table:
    """Keep the first ``n`` rows."""
    return _table(table.columns, table.rows[: max(0, n)])


@_op()
async def distinct(table: Table, columns: list[str] | None = None) -> Table:
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


# op → (a, b) → value. Arithmetic yields a number; comparisons yield 1/0 (a handy binary target).
# `/` guards divide-by-zero → None. Comparisons wrap in int() so True/False become 1/0.
_DERIVE_OPS: dict[str, Any] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b if b != 0 else None,
    ">": lambda a, b: int(a > b),
    ">=": lambda a, b: int(a >= b),
    "<": lambda a, b: int(a < b),
    "<=": lambda a, b: int(a <= b),
    "==": lambda a, b: int(a == b),
    "!=": lambda a, b: int(a != b),
}


@_op()
async def derive_column(
    table: Table,
    name: str,
    left: str,
    op: Literal["+", "-", "*", "/", ">", ">=", "<", "<=", "==", "!="],
    right: str,
) -> Table:
    """Add ``name`` = ``left`` <op> ``right``. Arithmetic (+,-,*,/) yields a number; comparisons
    (>,>=,<,<=,==,!=) yield 1/0 — handy for a binary target (e.g. ``revenue > 1000``). ``right`` is
    a numeric constant or another column name. For multi-branch conditionals, use ``sql``'s CASE WHEN.
    """
    _require_numeric(table, left, "derive_column")
    const: float | None = None
    try:
        const = float(right)
    except ValueError:
        _require_numeric(table, right, "derive_column")

    def compute(r: dict[str, Any]) -> Any:
        a = r[left]
        b = const if const is not None else r[right]
        return None if a is None or b is None else _DERIVE_OPS[op](a, b)

    cols = table.columns + ([name] if name not in table.columns else [])
    rows = [{**r, name: compute(r)} for r in table.rows]
    return _table(cols, rows)


@_op()
async def date_part(
    table: Table, column: str, part: Literal["year", "month", "day", "weekday"]
) -> Table:
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


@_op()
async def join(left: Table, right: Table, on: str, how: Literal["inner", "left"] = "inner") -> Table:
    """Join two tables on a shared column (a real SQL join via pandas). how ∈ inner, left.

    Unlike a lookup, duplicate keys on either side multiply matching rows (standard join
    semantics). Overlapping non-key columns from the right are suffixed ``_right``.
    """
    _need(left, on)
    _need(right, on)
    if how not in {"inner", "left"}:
        raise ValueError(f"join: how must be inner/left, got {how!r}")
    merged = _to_df(left).merge(_to_df(right), on=on, how=how, suffixes=("", "_right"))
    return _from_df(merged)


@_op()
async def sql(query: str, t1: Table, t2: Table | None = None, t3: Table | None = None) -> Table:
    """Run a DuckDB SQL query over the wired tables — the workhorse for filter/join/group/pivot/
    window in one step. Reference the inputs by port name: ``t1`` (the piped/primary input), and
    ``t2``/``t3`` if you wire them (e.g. ``SELECT ... FROM t1 JOIN t2 ON ...``). Add ``ORDER BY``
    for a stable result. Sandboxed: no file or network access (use ``read_csv`` to load data).
    """
    import duckdb

    con = duckdb.connect(config={"enable_external_access": False})
    try:
        for name, tbl in (("t1", t1), ("t2", t2), ("t3", t3)):
            if tbl is not None:
                con.register(name, _to_df(tbl))
        try:
            result = con.execute(query).fetchdf()
        except duckdb.Error as exc:
            raise ValueError(f"sql: query failed: {exc}") from exc
    finally:
        con.close()
    return _from_df(result)


# --- aggregate / analyze -------------------------------------------------------------------


@_op()
async def group_by(
    table: Table,
    keys: list[str],
    metric: str,
    aggs: list[Literal["sum", "mean", "min", "max", "count"]] | None = None,
) -> Table:
    """Group by ``keys`` (empty → overall total) and aggregate ``metric``. aggs ⊆ sum, mean, min,
    max, count (default ["sum"]). Output columns: keys + one per agg (``count`` is a row count).

    Grouping is done with pandas; the output schema (``count`` int, ``{metric}_{agg}`` rounded to
    4) is a stable contract the analysis blocks rely on.
    """
    import pandas as pd

    keys = _cols(keys)
    agg_list = _cols(aggs) or ["sum"]
    _need(table, *keys)
    unknown = [a for a in agg_list if a not in {*_NUMERIC_AGGS, "count"}]
    if unknown:
        raise ValueError(f"group_by: unknown aggs {unknown}; use sum/mean/min/max/count")
    numeric = [a for a in agg_list if a != "count"]
    if numeric:
        _require_numeric(table, metric, "group_by")

    df = _to_df(table)

    def agg_row(sub: Any) -> dict[str, Any]:
        row: dict[str, Any] = {}
        if "count" in agg_list:
            row["count"] = int(len(sub))
        series = pd.to_numeric(sub[metric], errors="coerce").dropna() if numeric else None
        for a in numeric:
            row[f"{metric}_{a}"] = round(float(getattr(series, a)()), 4) if len(series) else None
        return row

    out_rows: list[dict[str, Any]] = []
    if keys:
        for key_vals, sub in df.groupby(keys, dropna=False, sort=False):
            key_tuple = key_vals if isinstance(key_vals, tuple) else (key_vals,)
            row = {k: (None if pd.isna(v) else v) for k, v in zip(keys, key_tuple, strict=True)}
            row.update(agg_row(sub))
            out_rows.append(row)
    else:
        out_rows.append(agg_row(df))

    out_cols = [*keys, *(("count",) if "count" in agg_list else ()), *(f"{metric}_{a}" for a in numeric)]
    return _from_df(pd.DataFrame(out_rows, columns=out_cols))


@_op()
async def describe(table: Table, columns: list[str] | None = None) -> Table:
    """Summary stats per numeric column (empty → all numeric): count, mean, std, min, p25,
    median, p75, max.
    """
    import pandas as pd

    cols = _cols(columns) or [c for c in table.columns if _column_is_numeric(table, c)]
    _need(table, *cols)
    df = _to_df(table)
    # label column is "field" (not "column" — the latter is a SQL reserved word and would break a
    # downstream sql(SELECT column ...) step).
    out_cols = ["field", "count", "mean", "std", "min", "p25", "median", "p75", "max"]
    rows: list[dict[str, Any]] = []
    for c in cols:
        _require_numeric(table, c, "describe")
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        has = len(s) > 0
        rows.append({
            "field": c,
            "count": int(s.count()),
            "mean": round(float(s.mean()), 4) if has else None,
            "std": round(float(s.std(ddof=1)), 4) if len(s) > 1 else 0.0,
            "min": float(s.min()) if has else None,
            "p25": round(float(s.quantile(0.25)), 4) if has else None,
            "median": round(float(s.median()), 4) if has else None,
            "p75": round(float(s.quantile(0.75)), 4) if has else None,
            "max": float(s.max()) if has else None,
        })
    return _from_df(pd.DataFrame(rows, columns=out_cols))


@_op()
async def value_counts(table: Table, column: str) -> Table:
    """Frequency of each distinct value in ``column``, most frequent first."""
    _need(table, column)
    counts: dict[Any, int] = {}
    for r in table.rows:
        counts[r[column]] = counts.get(r[column], 0) + 1
    rows = [{"value": k, "count": v} for k, v in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)]
    return _table(["value", "count"], rows)


@_op()
async def top_n(table: Table, by: str, n: int = 10) -> Table:
    """The top ``n`` rows by ``by`` (descending; None-valued rows never count as top)."""
    _need(table, by)
    ranked = _sorted_non_null_first(table.rows, by, descending=True)
    return _table(table.columns, ranked[: max(0, n)])


# --- reshape / window / stats (pandas-backed) ----------------------------------------------


@_op()
async def pivot(
    table: Table,
    index: str,
    columns: str,
    values: str,
    aggfunc: Literal["sum", "mean", "min", "max", "count"] = "sum",
) -> Table:
    """Long→wide: one row per ``index``, one column per distinct ``columns`` value, cells are
    ``aggfunc`` of ``values``. aggfunc ∈ sum, mean, min, max, count.
    """
    import pandas as pd

    _need(table, index, columns, values)
    if aggfunc not in {*_NUMERIC_AGGS, "count"}:
        raise ValueError(f"pivot: aggfunc must be sum/mean/min/max/count, got {aggfunc!r}")
    pt = pd.pivot_table(
        _to_df(table), index=index, columns=columns, values=values, aggfunc=aggfunc, observed=True
    ).round(4)
    pt.columns = [str(c) for c in pt.columns]
    return _from_df(pt.reset_index())


@_op()
async def melt(
    table: Table, id_vars: list[str], value_vars: list[str] | None = None,
    var_name: str = "variable", value_name: str = "value",
) -> Table:
    """Wide→long: keep ``id_vars``, unpivot the rest (or ``value_vars``) into var/value columns."""
    ids = _cols(id_vars)
    vals = _cols(value_vars) or None
    _need(table, *ids, *(vals or []))
    return _from_df(
        _to_df(table).melt(id_vars=ids, value_vars=vals, var_name=var_name, value_name=value_name)
    )


@_op()
async def concat(top: Table, bottom: Table) -> Table:
    """Stack two tables' rows (union). Columns are the union; missing cells become null."""
    import pandas as pd

    return _from_df(pd.concat([_to_df(top), _to_df(bottom)], ignore_index=True))


@_op()
async def fill_missing(
    table: Table,
    column: str,
    method: Literal["value", "mean", "median", "mode", "ffill", "bfill"] = "value",
    value: Any = None,
) -> Table:
    """Fill missing values in ``column``. method ∈ value, mean, median, mode, ffill, bfill."""
    import pandas as pd

    _need(table, column)
    if method not in {"value", "mean", "median", "mode", "ffill", "bfill"}:
        raise ValueError(
            f"fill_missing: method must be value/mean/median/mode/ffill/bfill, got {method!r}"
        )
    if method == "value" and value is None:
        raise ValueError("fill_missing: method='value' needs a `value` (or use mean/median/mode/ffill/bfill)")
    df = _to_df(table)
    if method == "value":
        df[column] = df[column].fillna(value)
    elif method in {"mean", "median"}:
        num = pd.to_numeric(df[column], errors="coerce")
        df[column] = num.fillna(getattr(num, method)())
    elif method == "mode":
        modes = df[column].mode()  # can be empty (all-null) or multi-valued (ties) → take the first
        if len(modes):
            df[column] = df[column].fillna(modes.iloc[0])
    else:
        df[column] = df[column].ffill() if method == "ffill" else df[column].bfill()
    return _from_df(df)


@_op()
async def normalize(
    table: Table,
    column: str,
    method: Literal["minmax", "zscore"] = "minmax",
    name: str | None = None,
) -> Table:
    """Scale a numeric ``column`` in place (or into ``name``). minmax → ``(x−min)/(max−min)`` in
    [0, 1] (a constant column → all 0); zscore → ``(x−mean)/std`` with population std (ddof=0),
    matching scikit-learn's StandardScaler.
    """
    import pandas as pd

    _require_numeric(table, column, "normalize")
    if method not in {"minmax", "zscore"}:
        raise ValueError(f"normalize: method must be minmax/zscore, got {method!r}")
    df = _to_df(table)
    s = pd.to_numeric(df[column], errors="coerce")
    if method == "minmax":
        lo, hi = s.min(), s.max()
        scaled = (s - lo) / (hi - lo) if hi != lo else s * 0.0
    else:
        sd = s.std(ddof=0)
        scaled = (s - s.mean()) / sd if sd else s * 0.0
    df[name or column] = scaled
    return _from_df(df)


@_op()
async def encode(
    table: Table,
    column: str,
    method: Literal["label", "onehot"] = "label",
    name: str | None = None,
) -> Table:
    """Encode a categorical ``column``. label → integer codes ``0..k−1`` by sorted category value
    (matching scikit-learn's LabelEncoder), in place or into ``name``. onehot → append integer
    ``{column}_{value}`` indicator columns and drop the original.
    """
    import pandas as pd

    _need(table, column)
    df = _to_df(table)
    if method == "label":
        cats = sorted(df[column].dropna().unique(), key=str)
        mapping = {c: i for i, c in enumerate(cats)}
        df[name or column] = df[column].map(mapping).astype("Int64")
    elif method == "onehot":
        dummies = pd.get_dummies(df[column], prefix=column).astype(int)
        df = pd.concat([df.drop(columns=[column]), dummies], axis=1)
    else:
        raise ValueError(f"encode: method must be label/onehot, got {method!r}")
    return _from_df(df)


@_op()
async def rank(
    table: Table,
    by: str,
    name: str = "rank",
    descending: bool = True,
    method: Literal["dense", "min", "first", "average"] = "dense",
) -> Table:
    """Add a ``name`` column ranking rows by ``by`` (1 = top). method ∈ dense, min, first, average."""
    import pandas as pd

    _require_numeric(table, by, "rank")
    if method not in {"dense", "min", "first", "average"}:
        raise ValueError(f"rank: method must be dense/min/first/average, got {method!r}")
    df = _to_df(table)
    ranks = pd.to_numeric(df[by], errors="coerce").rank(ascending=not descending, method=method)
    df[name] = ranks.astype("Int64")
    return _from_df(df)


@_op(name="bin")
async def bin_column(table: Table, column: str, bins: int = 4, name: str | None = None) -> Table:
    """Bucket a numeric ``column`` into ``bins`` equal-width bins; adds a label column."""
    import pandas as pd

    _require_numeric(table, column, "bin")
    df = _to_df(table)
    out = name or f"{column}_bin"
    df[out] = pd.cut(pd.to_numeric(df[column], errors="coerce"), bins=bins).astype("str")
    return _from_df(df)


@_op()
async def correlation(
    table: Table,
    columns: list[str] | None = None,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
) -> Table:
    """Correlation matrix over numeric columns (a ``field`` label col + one col each). method ∈
    pearson, spearman, kendall. For a coefficient **with a p-value** on two columns, use ``corr_test``.
    """
    import pandas as pd

    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError(f"correlation: method must be pearson/spearman/kendall, got {method!r}")
    cols = _cols(columns) or [c for c in table.columns if _column_is_numeric(table, c)]
    _need(table, *cols)
    for c in cols:
        _require_numeric(table, c, "correlation")
    num = _to_df(table)[cols].apply(pd.to_numeric, errors="coerce")
    # label column "field" (not "column" — a SQL reserved word) so a downstream sql() can select it
    return _from_df(num.corr(method=method).round(4).reset_index(names="field"))


# --- inferential statistics (scipy.stats — a base dep via scikit-learn, lazy-imported) -----
# These ops report a test statistic and, where defined, a p-value — the analyst composes any
# significance verdict (e.g. |r|≥0.5 and p<0.05) itself; there is no built-in rubric. Pure &
# deterministic, so they cache normally.

_CORR_MIN_N = 3  # scipy.stats.pearsonr needs ≥3 pairs for a defined p-value
_SHAPIRO_MIN_N = 3
_NORMALTEST_MIN_N = 8  # D'Agostino-Pearson normaltest needs ≥8 samples
_ANDERSON_5PCT_LEVEL = 5.0  # Anderson-Darling significance level whose critical value we report
_TWO_GROUPS = 2  # ttest/welch/mannwhitney compare exactly two groups
_MIN_CATEGORIES = 2  # chi-square needs a ≥2×2 contingency table


@_op()
async def corr_test(
    table: Table,
    x: str,
    y: str,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
) -> Table:
    """Correlation between two numeric columns **with a significance test** — a 1-row table
    (``coefficient``, ``p_value``, ``n``). method ∈ pearson, spearman, kendall. Rows missing x or y
    are dropped; needs ≥3 complete pairs. End in ``to_markdown`` to report both, or slice one value
    into ``answer``.
    """
    import pandas as pd
    from scipy import stats

    _need(table, x, y)
    df = _to_df(table)[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < _CORR_MIN_N:
        raise ValueError(f"corr_test: need ≥{_CORR_MIN_N} complete (x, y) pairs, got {len(df)}")
    fn = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}[method]
    res = fn(df[x].to_numpy(), df[y].to_numpy())
    return _from_df(pd.DataFrame([{"coefficient": float(res[0]), "p_value": float(res[1]), "n": len(df)}]))


@_op()
async def normality_test(
    table: Table,
    column: str,
    method: Literal["shapiro", "normaltest", "anderson"] = "shapiro",
) -> Table:
    """Test whether a numeric ``column`` is normally distributed — a 1-row table (``statistic``,
    ``p_value``, ``critical_value``). shapiro/normaltest report ``p_value`` (reject normality if
    p < α); anderson reports ``statistic`` + the 5% ``critical_value`` (reject if statistic >
    critical_value). Missing values dropped; shapiro needs ≥3, normaltest ≥8.
    """
    import pandas as pd
    from scipy import stats

    _need(table, column)
    s = pd.to_numeric(_to_df(table)[column], errors="coerce").dropna().to_numpy()
    stat: float | None = None
    p_value: float | None = None
    critical: float | None = None
    if method == "shapiro":
        if len(s) < _SHAPIRO_MIN_N:
            raise ValueError(f"normality_test: shapiro needs ≥{_SHAPIRO_MIN_N} values, got {len(s)}")
        res = stats.shapiro(s)
        stat, p_value = float(res[0]), float(res[1])
    elif method == "normaltest":
        if len(s) < _NORMALTEST_MIN_N:
            raise ValueError(f"normality_test: normaltest needs ≥{_NORMALTEST_MIN_N} values, got {len(s)}")
        res = stats.normaltest(s)
        stat, p_value = float(res[0]), float(res[1])
    elif method == "anderson":
        res = stats.anderson(s, dist="norm")
        levels = [round(float(x), 1) for x in res.significance_level]
        idx = levels.index(_ANDERSON_5PCT_LEVEL) if _ANDERSON_5PCT_LEVEL in levels else 2
        stat, critical = float(res.statistic), float(res.critical_values[idx])
    else:
        raise ValueError(f"normality_test: method must be shapiro/normaltest/anderson, got {method!r}")
    return _from_df(pd.DataFrame([{"statistic": stat, "p_value": p_value, "critical_value": critical}]))


@_op()
async def group_test(
    table: Table,
    value: str,
    group: str,
    test: Literal["ttest", "welch", "mannwhitney", "anova", "kruskal"] = "ttest",
) -> Table:
    """Test whether numeric ``value`` differs across the categories in ``group`` — a 1-row table
    (``statistic``, ``p_value``, ``n_groups``). ttest (Student), welch (unequal-variance t-test) and
    mannwhitney (rank-sum) compare **exactly two** groups; anova (one-way F) and kruskal compare two
    or more. Rows missing ``value`` or ``group`` are dropped.
    """
    import pandas as pd
    from scipy import stats

    _need(table, value, group)
    df = _to_df(table)[[value, group]].copy()
    df[value] = pd.to_numeric(df[value], errors="coerce")
    df = df.dropna(subset=[value, group])
    samples = [g[value].to_numpy() for _, g in df.groupby(group) if len(g)]
    n_groups = len(samples)
    if test in {"ttest", "welch", "mannwhitney"} and n_groups != _TWO_GROUPS:
        raise ValueError(f"group_test: test={test!r} needs exactly {_TWO_GROUPS} groups, got {n_groups}")
    if test in {"anova", "kruskal"} and n_groups < _TWO_GROUPS:
        raise ValueError(f"group_test: test={test!r} needs ≥{_TWO_GROUPS} groups, got {n_groups}")
    if test == "ttest":
        res = stats.ttest_ind(samples[0], samples[1], equal_var=True)
    elif test == "welch":
        res = stats.ttest_ind(samples[0], samples[1], equal_var=False)
    elif test == "mannwhitney":
        res = stats.mannwhitneyu(samples[0], samples[1])
    elif test == "anova":
        res = stats.f_oneway(*samples)
    elif test == "kruskal":
        res = stats.kruskal(*samples)
    else:
        raise ValueError(f"group_test: test must be ttest/welch/mannwhitney/anova/kruskal, got {test!r}")
    return _from_df(pd.DataFrame([{"statistic": float(res[0]), "p_value": float(res[1]), "n_groups": n_groups}]))


@_op()
async def chi_square(table: Table, column1: str, column2: str) -> Table:
    """Chi-square test of independence between two categorical columns — a 1-row table
    (``statistic``, ``p_value``, ``dof``), from the contingency table of the two columns.
    """
    import pandas as pd
    from scipy import stats

    _need(table, column1, column2)
    df = _to_df(table)[[column1, column2]].dropna()
    ct = pd.crosstab(df[column1], df[column2])
    if ct.shape[0] < _MIN_CATEGORIES or ct.shape[1] < _MIN_CATEGORIES:
        raise ValueError(
            f"chi_square: need ≥{_MIN_CATEGORIES} categories in each column, got table shape {ct.shape}"
        )
    chi2, p_value, dof, _ = stats.chi2_contingency(ct)
    return _from_df(pd.DataFrame([{"statistic": float(chi2), "p_value": float(p_value), "dof": int(dof)}]))


@_op()
async def pct_change(table: Table, column: str, name: str | None = None) -> Table:
    """Row-over-row percent change of a numeric ``column`` (adds ``{column}_pct_change``)."""
    import pandas as pd

    _require_numeric(table, column, "pct_change")
    df = _to_df(table)
    out = name or f"{column}_pct_change"
    df[out] = (pd.to_numeric(df[column], errors="coerce").pct_change() * 100).round(4)
    return _from_df(df)


@_op()
async def rolling(
    table: Table,
    column: str,
    window: int,
    name: str | None = None,
    stat: Literal["mean", "sum", "min", "max"] = "mean",
) -> Table:
    """Rolling-window ``stat`` over a numeric ``column`` (adds ``{column}_rolling_{stat}``).
    stat ∈ mean, sum, min, max.
    """
    import pandas as pd

    _require_numeric(table, column, "rolling")
    if stat not in {"mean", "sum", "min", "max"}:
        raise ValueError(f"rolling: stat must be mean/sum/min/max, got {stat!r}")
    df = _to_df(table)
    out = name or f"{column}_rolling_{stat}"
    windowed = pd.to_numeric(df[column], errors="coerce").rolling(window)
    df[out] = getattr(windowed, stat)().round(4)
    return _from_df(df)


@_op()
async def quantile(table: Table, column: str, q: float) -> Table:
    """The ``q``-quantile (``q`` in [0, 1]) of a numeric ``column`` — a 1×1 table (``quantile``)."""
    import pandas as pd

    _require_numeric(table, column, "quantile")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"quantile: q must be in [0, 1], got {q}")
    s = pd.to_numeric(_to_df(table)[column], errors="coerce").dropna()
    val = s.quantile(q) if len(s) else float("nan")
    return _from_df(pd.DataFrame([{"quantile": val}]).round(4))


@_op()
async def outliers(
    table: Table,
    column: str,
    method: Literal["iqr", "zscore"] = "iqr",
    factor: float | None = None,
) -> Table:
    """Outliers of a numeric ``column`` — a 1-row summary: ``lower``, ``upper`` (the fences),
    ``count`` (values outside them), ``total``. method ∈ iqr (fences Q1−k·IQR / Q3+k·IQR) or zscore
    (mean ± k·std, population ddof=0). ``factor`` is k — defaults to 1.5 (iqr) or 3.0 (zscore).
    """
    import pandas as pd

    if method not in {"iqr", "zscore"}:
        raise ValueError(f"outliers: method must be iqr/zscore, got {method!r}")
    _require_numeric(table, column, "outliers")
    s = pd.to_numeric(_to_df(table)[column], errors="coerce").dropna()
    if s.empty:
        return _from_df(pd.DataFrame([{"lower": None, "upper": None, "count": 0, "total": 0}]))
    if method == "iqr":
        k = 1.5 if factor is None else factor
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        count = int(((s < lower) | (s > upper)).sum())
    else:
        k = 3.0 if factor is None else factor
        mean, sd = float(s.mean()), float(s.std(ddof=0))
        lower, upper = mean - k * sd, mean + k * sd
        count = int(((s < lower) | (s > upper)).sum()) if sd else 0
    row = {"lower": round(lower, 4), "upper": round(upper, 4), "count": count, "total": int(len(s))}
    return _from_df(pd.DataFrame([row]))


@_op()
async def distribution(table: Table, column: str) -> Table:
    """Shape of a numeric ``column`` — a 1-row table: mean, std, skewness, kurtosis (Fisher)."""
    import pandas as pd

    _require_numeric(table, column, "distribution")
    s = pd.to_numeric(_to_df(table)[column], errors="coerce").dropna()
    row = {
        "mean": s.mean(),
        "std": s.std(ddof=1) if len(s) > 1 else 0.0,
        "skewness": s.skew(),
        "kurtosis": s.kurt(),
    }
    return _from_df(pd.DataFrame([row]).round(4))


# --- machine learning (scikit-learn) -------------------------------------------------------
# scikit-learn is a base dependency; these ops still import it lazily *inside* the op bodies so
# importing this module for catalog rendering (every CLI startup) stays sklearn-free. Results are
# deterministic for a fixed ``random_state`` but depend on the scikit-learn version, which the recipe
# fingerprint does NOT capture — bump CACHE_VERSION when upgrading sklearn or changing ML semantics.
# The ops reproduce scikit-learn's defaults so the numbers match a default-based reference solution.

_ML_RANDOM_STATE = 42  # default seed; a concrete int (never None) so ML ops stay pure & cacheable
_CV_FOLDS = 5


def _require_sklearn() -> None:
    """Defensive guard — sklearn is a base dependency, so this should never trigger."""
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover - sklearn is a base dependency
        raise ValueError("ML operators need scikit-learn, which is a base dependency of vibe.") from exc


def _encode_features(x_df: Any, encode: str) -> Any:
    """Encode an X frame for an estimator. ``onehot`` → ``pd.get_dummies`` (categoricals only;
    numerics pass through). ``label`` → integer codes (sorted category) for object columns only,
    numerics untouched — so a question that "label-encodes the features" is reproduced.
    """
    import pandas as pd

    if encode == "label":
        out = x_df.copy()
        for col in out.columns:
            if not pd.api.types.is_numeric_dtype(out[col]):  # encode only non-numeric features
                cats = sorted(out[col].dropna().unique(), key=str)
                out[col] = out[col].map({c: i for i, c in enumerate(cats)})
        return out
    if encode == "onehot":
        return pd.get_dummies(x_df)
    raise ValueError(f"ml: encode must be onehot/label, got {encode!r}")


def _ml_xy(table: Table, target: str, features: list[str], encode: str = "onehot") -> tuple[Any, Any]:
    """Build (X, y): encode categorical features (``encode`` ∈ onehot|label), drop rows with any
    missing value.
    """
    feats = _cols(features)
    if not feats:
        raise ValueError("ml: `features` must list at least one column")
    _need(table, target, *feats)
    df = _to_df(table)[[*feats, target]].dropna()
    if len(df) <= 1:
        raise ValueError("ml: need at least 2 complete rows after dropping missing values")
    return _encode_features(df[feats], encode), df[target]


def _score_table(value: float) -> Table:
    """A 1×1 ``score`` table — kept at full precision; ``answer(decimals=…)`` does the rounding."""
    import pandas as pd

    return _from_df(pd.DataFrame([{"score": float(value)}]))


# scikit-learn estimators keyed by short name. Each entry is a thunk (built only when selected), and
# ``random_state`` is passed only to estimators whose constructor accepts it (LinearRegression, KNN,
# GaussianNB do not) — passing it blindly raises TypeError.
def _reg_models(random_state: int) -> dict[str, Any]:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor

    return {
        "linear": lambda: LinearRegression(),
        "ridge": lambda: Ridge(random_state=random_state),
        "lasso": lambda: Lasso(random_state=random_state),
        "tree": lambda: DecisionTreeRegressor(random_state=random_state),
        "rf": lambda: RandomForestRegressor(random_state=random_state),
        "gbr": lambda: GradientBoostingRegressor(random_state=random_state),
        "knn": lambda: KNeighborsRegressor(),
        "svr": lambda: SVR(),
    }


def _clf_models(random_state: int) -> dict[str, Any]:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    return {
        "logreg": lambda: LogisticRegression(max_iter=1000, random_state=random_state),
        "tree": lambda: DecisionTreeClassifier(random_state=random_state),
        "rf": lambda: RandomForestClassifier(random_state=random_state),
        "gbm": lambda: GradientBoostingClassifier(random_state=random_state),
        "knn": lambda: KNeighborsClassifier(),
        "svc": lambda: SVC(random_state=random_state),
        "nb": lambda: GaussianNB(),
    }


def _reg_scorer(metric: str):  # noqa: ANN202 (callable (y_true, y_pred) -> float)
    from sklearn.metrics import (
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
        r2_score,
    )

    return {
        "r2": r2_score,
        "mse": mean_squared_error,
        "rmse": lambda yt, yp: mean_squared_error(yt, yp) ** 0.5,
        "mae": mean_absolute_error,
        "mape": mean_absolute_percentage_error,
    }[metric]


def _clf_scorer(metric: str):  # noqa: ANN202 (callable (y_true, y_pred) -> float)
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    if metric == "accuracy":
        return accuracy_score
    fn = {"f1": f1_score, "precision": precision_score, "recall": recall_score}[metric]
    return lambda yt, yp: fn(yt, yp, average="weighted", zero_division=0)


# cross_val_score needs a scoring *string* and returns errors negated (neg_*), so we map + un-negate.
_REG_CV_SCORING = {
    "r2": "r2", "mse": "neg_mean_squared_error", "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error", "mape": "neg_mean_absolute_percentage_error",
}
_CLF_CV_SCORING = {
    "accuracy": "accuracy", "f1": "f1_weighted",
    "precision": "precision_weighted", "recall": "recall_weighted",
}


def _evaluate_estimator(
    make_est: Any, x: Any, y: Any, *, evaluate: str, scorer: Any, cv_scoring: str,
    test_size: float, random_state: int, scale: bool, cv_folds: int,
) -> float:
    """Fit/score ``make_est()`` under one of three regimes: ``holdout`` (train/test split, score on
    test), ``full`` (fit on all rows, score on the same — the "no split stated" / training-metric
    case), or ``cv`` (k-fold cross_val_score mean). Optional StandardScaler is fit on the train fold
    only (via a Pipeline under cv) so scaling never leaks.
    """
    import pandas as pd
    from sklearn.model_selection import cross_val_score, train_test_split

    if evaluate == "cv":
        est = make_est()
        if scale:
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            est = make_pipeline(StandardScaler(), est)
        folds = max(2, min(cv_folds, len(y)))
        scores = cross_val_score(est, x, y, cv=folds, scoring=cv_scoring)
        val = float(scores.mean())
        return -val if cv_scoring.startswith("neg_") else val

    if evaluate == "full":
        x_tr, x_te, y_tr, y_te = x, x, y, y
    elif evaluate == "holdout":
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=test_size, random_state=random_state)
    else:
        raise ValueError(f"ml: evaluate must be holdout/full/cv, got {evaluate!r}")

    if scale:
        from sklearn.preprocessing import StandardScaler

        sc = StandardScaler().fit(x_tr)
        x_tr = pd.DataFrame(sc.transform(x_tr), index=x_tr.index, columns=x_tr.columns)
        x_te = pd.DataFrame(sc.transform(x_te), index=x_te.index, columns=x_te.columns)
    est = make_est()
    est.fit(x_tr, y_tr)
    return float(scorer(y_te, est.predict(x_te)))


@_op()
async def ml_regression(  # noqa: PLR0913, PLR0917 (typed ML knobs; executor binds by keyword)
    table: Table,
    target: str,
    features: list[str],
    model: Literal["linear", "ridge", "lasso", "tree", "rf", "gbr", "knn", "svr"] = "linear",
    metric: Literal["r2", "rmse", "mse", "mae", "mape"] = "r2",
    evaluate: Literal["holdout", "full", "cv"] = "holdout",
    test_size: float = 0.2,
    random_state: int = _ML_RANDOM_STATE,
    scale: bool = False,
    cv_folds: int = _CV_FOLDS,
    encode: Literal["onehot", "label"] = "onehot",
) -> Table:
    """Fit a regression model and return a 1×1 ``score`` table (reproduces scikit-learn defaults).
    ``evaluate``: ``holdout`` (train/test split, score on test), ``full`` (fit on all rows, score on
    the same — use when a question states no split), ``cv`` (k-fold mean). Set ``random_state`` /
    ``test_size`` to match the question; ``scale`` standardizes features (fit on train) for knn/svr.
    model ∈ linear|ridge|lasso|tree|rf|gbr|knn|svr; metric ∈ r2|rmse|mse|mae|mape. End with
    ``answer(decimals=…)`` at the asked precision. Categorical features one-hot encoded, missing rows
    dropped. Deterministic for a given ``random_state``.
    """
    _require_sklearn()
    import pandas as pd

    models = _reg_models(random_state)
    if model not in models:
        raise ValueError(f"ml_regression: model must be one of {sorted(models)}, got {model!r}")
    x, y = _ml_xy(table, target, features, encode=encode)
    y = pd.to_numeric(y, errors="coerce")
    score = _evaluate_estimator(
        models[model], x, y, evaluate=evaluate, scorer=_reg_scorer(metric),
        cv_scoring=_REG_CV_SCORING[metric], test_size=test_size, random_state=random_state,
        scale=scale, cv_folds=cv_folds,
    )
    return _score_table(score)


@_op()
async def ml_classification(  # noqa: PLR0913, PLR0917 (typed ML knobs; executor binds by keyword)
    table: Table,
    target: str,
    features: list[str],
    model: Literal["logreg", "tree", "rf", "gbm", "knn", "svc", "nb"] = "logreg",
    metric: Literal["accuracy", "f1", "precision", "recall"] = "accuracy",
    evaluate: Literal["holdout", "full", "cv"] = "holdout",
    test_size: float = 0.2,
    random_state: int = _ML_RANDOM_STATE,
    scale: bool = False,
    cv_folds: int = _CV_FOLDS,
    encode: Literal["onehot", "label"] = "onehot",
) -> Table:
    """Fit a classifier and return a 1×1 ``score`` table (reproduces scikit-learn defaults).
    ``evaluate``: ``holdout`` (train/test split, score on test), ``full`` (fit + score on all rows),
    ``cv`` (k-fold mean). Set ``random_state`` / ``test_size`` to match the question; ``scale``
    standardizes features (fit on train) for knn/svc. model ∈ logreg|tree|rf|gbm|knn|svc|nb;
    metric ∈ accuracy|f1|precision|recall (f1/precision/recall are weighted). End with
    ``answer(decimals=…)``. Categorical features one-hot encoded, missing rows dropped. Deterministic
    for a given ``random_state``.
    """
    _require_sklearn()

    models = _clf_models(random_state)
    if model not in models:
        raise ValueError(f"ml_classification: model must be one of {sorted(models)}, got {model!r}")
    x, y = _ml_xy(table, target, features, encode=encode)
    score = _evaluate_estimator(
        models[model], x, y, evaluate=evaluate, scorer=_clf_scorer(metric),
        cv_scoring=_CLF_CV_SCORING[metric], test_size=test_size, random_state=random_state,
        scale=scale, cv_folds=cv_folds,
    )
    return _score_table(score)


@_op()
async def ml_cluster(
    table: Table,
    features: list[str],
    k: int,
    metric: Literal["silhouette", "inertia"] = "silhouette",
    random_state: int = _ML_RANDOM_STATE,
    encode: Literal["onehot", "label"] = "onehot",
) -> Table:
    """K-means over ``features`` (categoricals encoded per ``encode``, missing rows dropped) — a 1×1
    table (``score``): ``silhouette`` (cohesion/separation, higher is better) or ``inertia``.
    Deterministic for a given ``random_state``.
    """
    _require_sklearn()
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if metric not in {"silhouette", "inertia"}:
        raise ValueError(f"ml_cluster: metric must be silhouette/inertia, got {metric!r}")
    feats = _cols(features)
    _need(table, *feats)
    x = _encode_features(_to_df(table)[feats].dropna(), encode)
    if len(x) <= k:
        raise ValueError(f"ml_cluster: need more rows ({len(x)}) than clusters (k={k})")
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(x)
    score = silhouette_score(x, km.labels_) if metric == "silhouette" else km.inertia_
    return _score_table(score)


def _fi_estimator(model: str, task: str, random_state: int):  # noqa: ANN202 (sklearn estimator, lazy)
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    clf = task == "classification"
    if model == "rf":
        return (RandomForestClassifier if clf else RandomForestRegressor)(random_state=random_state)
    if model == "tree":
        return (DecisionTreeClassifier if clf else DecisionTreeRegressor)(random_state=random_state)
    if model == "linear":
        if clf:
            raise ValueError("feature_importance: model 'linear' needs a regression target (use logreg/rf/tree)")
        return LinearRegression()
    if not clf:
        raise ValueError("feature_importance: model 'logreg' needs a classification target (use linear/rf/tree)")
    return LogisticRegression(max_iter=1000, random_state=random_state)


@_op()
async def feature_importance(
    table: Table,
    target: str,
    features: list[str],
    model: Literal["rf", "tree", "linear", "logreg"] = "rf",
    task: Literal["auto", "classification", "regression"] = "auto",
    random_state: int = _ML_RANDOM_STATE,
    encode: Literal["onehot", "label"] = "onehot",
) -> Table:
    """Rank ``features`` by how much they predict ``target`` — a table of ``field, importance``
    (descending). rf/tree use impurity importances, linear/logreg use ``|coef|``. ``task=auto``
    infers regression for a numeric target, else classification. One-hot columns are summed back to
    their source feature, so the ranking is by the columns you named. Deterministic for ``random_state``.
    """
    _require_sklearn()
    import numpy as np
    import pandas as pd

    feats = _cols(features)
    resolved = task if task != "auto" else ("regression" if _column_is_numeric(table, target) else "classification")
    x, y = _ml_xy(table, target, feats, encode=encode)
    y_fit = pd.to_numeric(y, errors="coerce") if resolved == "regression" else y.astype("str")
    est = _fi_estimator(model, resolved, random_state)
    est.fit(x, y_fit)
    if hasattr(est, "feature_importances_"):
        imp = np.abs(np.asarray(est.feature_importances_))
    else:
        coef = np.asarray(est.coef_)
        imp = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    # sum one-hot-encoded columns (`{feature}_{value}`) back to the source feature the agent named
    totals = {f: 0.0 for f in feats}
    for col, value in zip(x.columns, imp, strict=True):
        src = next((f for f in feats if col == f or str(col).startswith(f + "_")), None)
        if src is not None:
            totals[src] += float(value)
    rows = sorted(
        ({"field": f, "importance": round(v, 4)} for f, v in totals.items()),
        key=lambda r: r["importance"],
        reverse=True,
    )
    return _table(["field", "importance"], rows)


@_op()
async def ml_predict(  # noqa: PLR0914, PLR0913, PLR0917 (typed ML knobs; executor binds by keyword)
    table: Table,
    target: str,
    features: list[str],
    model: Literal[
        "linear", "ridge", "lasso", "tree", "rf", "gbr", "knn", "svr",
        "logreg", "gbm", "svc", "nb",
    ] = "linear",
    task: Literal["regression", "classification"] = "regression",
    evaluate: Literal["holdout", "full"] = "holdout",
    on: Literal["test", "all"] = "test",
    test_size: float = 0.2,
    random_state: int = _ML_RANDOM_STATE,
    scale: bool = False,
    encode: Literal["onehot", "label"] = "onehot",
) -> Table:
    """Fit a model and return **per-row predictions** — a table of ``[*features, target, {target}_pred]``.
    ``evaluate`` fits on the train split (``holdout``) or all rows (``full``); ``on`` predicts the held-out
    ``test`` rows or ``all`` rows. Use for "predict …" questions, then ``filter_rows``/``answer`` to read a
    value. model must match ``task`` (regression: linear|ridge|lasso|tree|rf|gbr|knn|svr; classification:
    logreg|tree|rf|gbm|knn|svc|nb). Categorical features encoded per ``encode``, missing rows dropped.
    """
    _require_sklearn()
    import pandas as pd
    from sklearn.model_selection import train_test_split

    feats = _cols(features)
    if not feats:
        raise ValueError("ml_predict: `features` must list at least one column")
    _need(table, target, *feats)
    models = _reg_models(random_state) if task == "regression" else _clf_models(random_state)
    if model not in models:
        raise ValueError(
            f"ml_predict: model {model!r} is not valid for task={task!r}; choose one of {sorted(models)}"
        )
    df = _to_df(table)[[*feats, target]].dropna()
    if len(df) <= 1:
        raise ValueError("ml_predict: need at least 2 complete rows after dropping missing values")
    x = _encode_features(df[feats], encode)
    y = pd.to_numeric(df[target], errors="coerce") if task == "regression" else df[target]

    if evaluate == "full":
        x_fit, y_fit, pred_idx = x, y, x.index
    else:
        x_tr, x_te, y_tr, _ = train_test_split(x, y, test_size=test_size, random_state=random_state)
        x_fit, y_fit = x_tr, y_tr
        pred_idx = x.index if on == "all" else x_te.index

    x_model = x
    if scale:
        from sklearn.preprocessing import StandardScaler

        sc = StandardScaler().fit(x_fit)
        x_fit = pd.DataFrame(sc.transform(x_fit), index=x_fit.index, columns=x_fit.columns)
        x_model = pd.DataFrame(sc.transform(x), index=x.index, columns=x.columns)

    est = models[model]()
    est.fit(x_fit, y_fit)
    preds = est.predict(x_model.loc[pred_idx])

    pred_col = f"{target}_pred"
    rows: list[dict[str, Any]] = []
    for pos, idx in enumerate(pred_idx):
        src = df.loc[idx]
        p = preds[pos]
        pred_val = float(p) if task == "regression" else (p.item() if hasattr(p, "item") else p)
        rows.append({**{f: src[f] for f in feats}, target: src[target], pred_col: pred_val})
    return _table([*feats, target, pred_col], rows)


# --- insight (LLM-backed) ------------------------------------------------------------------
# These ops call the session model via an executor-injected `llm` caller (reserved param). Results
# are cached per recipe (goal/labels + input fingerprint), so a re-run is a cache hit — change the
# goal or `/graph cache clear` to regenerate. They need a live session (headless → clear error).

_MAX_NARRATE_ROWS = 50
_MAX_CLASSIFY_ROWS = 200


def _table_text(table: Table, max_rows: int) -> tuple[str, bool]:
    """A compact ``col | col`` rendering of up to ``max_rows`` rows; also whether it was truncated."""
    rows = table.rows[:max_rows]
    header = " | ".join(table.columns)
    body = "\n".join(" | ".join(str(r.get(c, "")) for c in table.columns) for r in rows)
    return f"{header}\n{body}", len(table.rows) > max_rows


@_op(needs_llm=True)
async def narrate(
    table: Table, goal: str = "", max_rows: int = _MAX_NARRATE_ROWS, llm: LLMCaller | None = None
) -> Report:
    """Write a short, factual prose takeaway about ``table`` (optionally focused on ``goal``) — an
    LLM step returning a Report. Best on a small summary/aggregate table (up to ``max_rows`` rows
    are shown). Cached per (goal + table); calls the model once.
    """
    assert llm is not None  # guaranteed by the executor for needs_llm ops
    data_text, truncated = _table_text(table, max_rows)
    focus = f" Focus on: {goal}." if goal else ""
    note = f"\n(showing first {max_rows} of {len(table.rows)} rows)" if truncated else ""
    prompt = (
        f"Interpret this table and write a concise, factual takeaway in 2-4 sentences.{focus}\n"
        "Ground every claim in the table: quote the exact figures and column names you cite, and "
        "invent nothing — if the data doesn't show it, don't say it.\n\n"
        f"{data_text}{note}"
    )
    text = await llm(prompt, system="You are a precise data analyst.", max_tokens=512)
    return Report(markdown=text.strip())


@_op(needs_llm=True)
async def classify(
    table: Table,
    column: str,
    labels: list[str],
    max_rows: int = _MAX_CLASSIFY_ROWS,
    llm: LLMCaller | None = None,
) -> Table:
    """Label each row by its ``column`` text with exactly one of ``labels`` — one batched LLM call;
    adds a ``{column}_label`` column. Errors (rather than silently truncating) if the table has more
    than ``max_rows`` rows — filter/aggregate first. Cached per (column + labels + table).
    """
    import json

    assert llm is not None  # guaranteed by the executor for needs_llm ops
    _need(table, column)
    labels = _cols(labels)
    if not labels:
        raise ValueError("classify: labels must be non-empty")
    if len(table.rows) > max_rows:
        raise ValueError(
            f"classify: {len(table.rows)} rows exceeds max_rows={max_rows}; filter or aggregate first"
        )
    values = [str(r.get(column, "")) for r in table.rows]
    numbered = "\n".join(f"{i}: {v}" for i, v in enumerate(values))
    prompt = (
        f"Classify each item into exactly one of these labels: {labels}.\n"
        f"Return ONLY a JSON array of {len(values)} strings — one label per item, in order, "
        f"no prose.\n\nItems:\n{numbered}"
    )
    raw = await llm(prompt, system="You label data. Output only a JSON array of labels.", max_tokens=2048)
    try:
        parsed = json.loads(raw[raw.index("[") : raw.rindex("]") + 1])
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"classify: model did not return a JSON array: {raw[:200]!r}") from exc
    if not isinstance(parsed, list) or len(parsed) != len(values):
        raise ValueError(f"classify: expected {len(values)} labels, got {len(parsed) if isinstance(parsed, list) else '?'}")
    allowed = set(labels)
    bad = sorted({str(p) for p in parsed if p not in allowed})
    if bad:
        raise ValueError(f"classify: model returned labels outside {labels}: {bad[:5]}")
    out_col = f"{column}_label"
    cols = table.columns + ([out_col] if out_col not in table.columns else [])
    rows = [{**r, out_col: str(parsed[i])} for i, r in enumerate(table.rows)]
    return _table(cols, rows)


# --- data-quality gates --------------------------------------------------------------------


@_op()
async def expect_columns(table: Table, columns: list[str]) -> Table:
    """Assert the table has the named columns; pass it through unchanged, else fail with a
    clear error the agent can act on.
    """
    missing = [c for c in _cols(columns) if c not in table.columns]
    if missing:
        raise ValueError(f"expect_columns: missing {missing}; available columns are {table.columns}")
    return table


@_op()
async def expect_no_nulls(table: Table, columns: list[str] | None = None) -> Table:
    """Assert no missing (None/blank) values in ``columns`` (or all); pass through unchanged."""
    check = _cols(columns) or table.columns
    _need(table, *check)
    for c in check:
        bad = sum(1 for r in table.rows if r[c] is None or r[c] == "")
        if bad:
            raise ValueError(f"expect_no_nulls: column {c!r} has {bad} missing value(s)")
    return table


@_op()
async def expect_unique(table: Table, columns: list[str]) -> Table:
    """Assert the given columns form a unique key; pass through unchanged."""
    keys = _cols(columns)
    _need(table, *keys)
    seen: set[tuple[Any, ...]] = set()
    dups = 0
    for r in table.rows:
        sig = tuple(r[c] for c in keys)
        dups += sig in seen
        seen.add(sig)
    if dups:
        raise ValueError(f"expect_unique: {keys} is not unique ({dups} duplicate row(s))")
    return table


# --- sinks (write a file, return a small handle) -------------------------------------------


@_op()
async def to_csv(table: Table, path: str) -> ExportResult:
    """Write the table to a CSV file at ``path``; returns a handle (path + shape), not the data.

    The write is reviewed at the graph_patch approval gate (the path shows in the diff). Cached by
    recipe — a re-run with the same inputs won't rewrite, and deleting the file won't regenerate
    it under a cache hit.
    """
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    _to_df(table).to_csv(p, index=False)
    return ExportResult(path=str(p), rows=len(table.rows), columns=list(table.columns))


def _mpl():  # noqa: ANN202 (matplotlib.pyplot + pandas, imported lazily)
    """Lazy matplotlib (headless ``Agg``) + pyplot + pandas — kept out of module import for startup."""
    import matplotlib

    matplotlib.use("Agg")  # headless, deterministic; no display backend
    import matplotlib.pyplot as plt
    import pandas as pd

    return plt, pd


def _save_fig(fig, path: str, kind: str) -> ChartResult:  # noqa: ANN001 (matplotlib Figure)
    """Layout + write ``fig`` to a PNG at ``path`` (always closing it); return a small handle."""
    import matplotlib.pyplot as plt

    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout()
        fig.savefig(p)
    finally:
        plt.close(fig)
    return ChartResult(path=str(p), kind=kind)


@_op()
async def chart(  # noqa: PLR0913, PLR0917, PLR0912, PLR0915 (one dispatch op over 7 chart kinds)
    table: Table,
    kind: Literal["bar", "line", "scatter", "histogram", "box", "pie", "heatmap"],
    path: str,
    x: str = "",
    y: str = "",
    column: str = "",
    by: str = "",
    series: str = "",
    labels: str = "",
    values: str = "",
    bins: int = 20,
    max_slices: int = 8,
    title: str = "",
) -> ChartResult:
    """Render a chart to a PNG at ``path``; returns a small handle (image bytes stay out of context).
    Required params per ``kind``: **bar/line/scatter** need ``x`` + ``y`` (line adds optional
    ``series`` for one line per value); **histogram** needs ``column`` (+ ``bins``); **box** needs
    ``column`` (+ optional ``by``); **pie** needs ``labels`` + ``values`` (+ ``max_slices``);
    **heatmap** plots the table's numeric matrix (e.g. a ``correlation``/``pivot`` output).
    """
    plt, pd = _mpl()
    df = _to_df(table)

    if kind in {"bar", "line", "scatter"}:
        if not x or not y:
            raise ValueError(f"chart: kind={kind!r} needs both x and y")
        if kind == "scatter":
            _require_numeric(table, x, "chart")
            _require_numeric(table, y, "chart")
            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
            ax.scatter(pd.to_numeric(df[x], errors="coerce"), pd.to_numeric(df[y], errors="coerce"), alpha=0.7)
            ax.set_title(title or f"{y} vs {x}"), ax.set_xlabel(x), ax.set_ylabel(y)
        else:
            _need(table, x, y)
            fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
            if kind == "line" and series:
                _need(table, series)
                for key, grp in df.groupby(series):
                    ax.plot(grp[x].astype("str"), pd.to_numeric(grp[y], errors="coerce"), label=str(key))
                ax.legend(title=series)
            elif kind == "line":
                ax.plot(df[x].astype("str"), pd.to_numeric(df[y], errors="coerce"))
            else:
                ax.bar(df[x].astype("str"), pd.to_numeric(df[y], errors="coerce"))
            ax.set_title(title), ax.set_xlabel(x), ax.set_ylabel(y)
            fig.autofmt_xdate()
    elif kind in {"histogram", "box"}:
        if not column:
            raise ValueError(f"chart: kind={kind!r} needs a column")
        _require_numeric(table, column, "chart")
        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
        if kind == "histogram":
            ax.hist(pd.to_numeric(df[column], errors="coerce").dropna(), bins=bins)
            ax.set_title(title or f"Distribution of {column}"), ax.set_xlabel(column), ax.set_ylabel("count")
        elif by:
            _need(table, by)
            groups, box_labels = [], []
            for key, grp in df.groupby(by):
                vals = pd.to_numeric(grp[column], errors="coerce").dropna()
                if len(vals):
                    groups.append(vals)
                    box_labels.append(str(key))
            ax.boxplot(groups, tick_labels=box_labels)
            ax.set_title(title or f"{column} distribution"), ax.set_xlabel(by), ax.set_ylabel(column)
        else:
            ax.boxplot(pd.to_numeric(df[column], errors="coerce").dropna())
            ax.set_title(title or f"{column} distribution"), ax.set_ylabel(column)
    elif kind == "pie":
        if not labels or not values:
            raise ValueError("chart: kind='pie' needs labels and values")
        _need(table, labels)
        _require_numeric(table, values, "chart")
        ser = (
            pd.to_numeric(df[values], errors="coerce")
            .groupby(df[labels].astype("str"))
            .sum()
            .sort_values(ascending=False)
        )
        if len(ser) > max_slices:
            ser = pd.concat([ser.iloc[:max_slices], pd.Series({"other": ser.iloc[max_slices:].sum()})])
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        ax.pie(ser.to_numpy(), labels=[str(i) for i in ser.index], autopct="%1.1f%%")
        ax.set_title(title or f"{values} by {labels}")
    else:  # heatmap — plots the numeric matrix; a leading non-numeric column labels the rows
        numeric = [c for c in table.columns if _column_is_numeric(table, c)]
        if not numeric:
            raise ValueError("chart: kind='heatmap' needs numeric columns to plot")
        label_col = next((c for c in table.columns if c not in numeric), None)
        mat = df[numeric].apply(pd.to_numeric, errors="coerce").to_numpy()
        row_labels = df[label_col].astype("str").tolist() if label_col else [str(i) for i in range(len(df))]
        fig, ax = plt.subplots(figsize=(1 + 0.7 * len(numeric), 1 + 0.5 * len(row_labels)), dpi=100)
        im = ax.imshow(mat, aspect="auto", cmap="coolwarm")
        ax.set_xticks(range(len(numeric)))
        ax.set_xticklabels(numeric, rotation=45, ha="right")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        for i in range(len(row_labels)):
            for j in range(len(numeric)):
                if mat[i][j] == mat[i][j]:  # skip NaN
                    ax.text(j, i, f"{mat[i][j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax)
        ax.set_title(title or "Heatmap")
    return _save_fig(fig, path, kind)


# --- report --------------------------------------------------------------------------------


@_op()
async def to_markdown(table: Table, title: str = "Report", max_rows: int = 50) -> Report:
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


@_op()
async def answer(table: Table, decimals: int | None = None) -> Report:
    """State a single result. Takes a **1×1 table** (one column, one row) and returns a report whose
    text is exactly that value — rounded to ``decimals`` if given (numeric only). Use as the final
    step of a "what is X?" pipeline so the answer is unambiguous; errors if the table isn't 1×1.
    """
    if len(table.columns) != 1 or len(table.rows) != 1:
        raise ValueError(
            f"answer: expected a 1×1 table (one column, one row), got {len(table.rows)} row(s) × "
            f"{len(table.columns)} column(s) — reduce to a single value first "
            "(e.g. a scalar sql/quantile/ml_* step)."
        )
    value = table.rows[0][table.columns[0]]
    if decimals is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
        value = round(float(value), decimals)
    return Report(markdown=str(value))


# --- classic-analysis blocks (the workflows, encoded) --------------------------------------


def _quick_profile() -> Graph:
    g = Graph()
    g.add(Node(id="d", op="describe", params={"columns": []}))
    g.add(Node(id="r", op="to_markdown", params={"max_rows": 50}, inputs={"table": "d"}))
    return g


def _rank_by() -> Graph:
    g = Graph()
    g.add(Node(id="g", op="group_by", params={"keys": [], "metric": "", "aggs": ["sum"]}))
    # top_n ranks by the summed metric — a derived column name resolved from the `metric` param.
    g.add(Node(id="t", op="top_n", params={"by": "{metric}_sum", "n": 5}, inputs={"table": "g"}))
    g.add(Node(id="r", op="to_markdown", params={"max_rows": 50}, inputs={"table": "t"}))
    return g


def _trend_by_period() -> Graph:
    g = Graph()
    g.add(Node(id="dp", op="date_part", params={"column": "", "part": "month"}))
    # group + sort by the derived date-part column ({date_column}_{period}), resolved at expand.
    g.add(Node(id="g", op="group_by",
               params={"keys": ["{date_column}_{period}"], "metric": "", "aggs": ["sum"]},
               inputs={"table": "dp"}))
    g.add(Node(id="s", op="sort_rows", params={"by": "{date_column}_{period}", "descending": False},
               inputs={"table": "g"}))
    g.add(Node(id="r", op="to_markdown", params={"max_rows": 100}, inputs={"table": "s"}))
    return g


def _month_over_month_growth() -> Graph:
    g = Graph()
    g.add(Node(id="dp", op="date_part", params={"column": "", "part": "month"}))
    g.add(Node(id="g", op="group_by",
               params={"keys": ["{date_column}_month"], "metric": "", "aggs": ["sum"]},
               inputs={"table": "dp"}))
    g.add(Node(id="s", op="sort_rows", params={"by": "{date_column}_month", "descending": False},
               inputs={"table": "g"}))
    g.add(Node(id="pc", op="pct_change", params={"column": "{metric}_sum"}, inputs={"table": "s"}))
    g.add(Node(id="r", op="to_markdown", params={"max_rows": 100}, inputs={"table": "pc"}))
    return g


def _segment_summary() -> Graph:
    g = Graph()
    g.add(Node(id="g", op="group_by", params={"keys": [], "metric": "", "aggs": ["count", "sum", "mean"]}))
    g.add(Node(id="r", op="to_markdown", params={"max_rows": 100}, inputs={"table": "g"}))
    return g


def _correlation_report() -> Graph:
    g = Graph()
    g.add(Node(id="c", op="correlation", params={"columns": []}))
    g.add(Node(id="r", op="to_markdown", params={"max_rows": 50}, inputs={"table": "c"}))
    return g


def _frequency_report() -> Graph:
    g = Graph()
    g.add(Node(id="v", op="value_counts", params={"column": ""}))
    g.add(Node(id="r", op="to_markdown", params={"max_rows": 50}, inputs={"table": "v"}))
    return g


_BLOCKS = [
    BlockDef(
        name="quick_profile", graph=_quick_profile(), library=_LIB,
        description="summary stats for every numeric column",
        input_ports={"table": ("d", "table")}, params={"title": ("r", "title")}, output="r",
    ),
    BlockDef(
        name="rank_by", graph=_rank_by(), library=_LIB,
        description="top N groups by a summed metric",
        input_ports={"table": ("g", "table")},
        params={"group_key": ("g", "keys"), "metric": ("g", "metric"),
                "n": ("t", "n"), "title": ("r", "title")},
        output="r",
    ),
    BlockDef(
        name="trend_by_period", graph=_trend_by_period(), library=_LIB,
        description="metric summed per calendar period (time series)",
        input_ports={"table": ("dp", "table")},
        params={"date_column": ("dp", "column"), "period": ("dp", "part"),
                "metric": ("g", "metric"), "title": ("r", "title")},
        output="r",
    ),
    BlockDef(
        name="month_over_month_growth", graph=_month_over_month_growth(), library=_LIB,
        description="metric summed per month with row-over-row % change",
        input_ports={"table": ("dp", "table")},
        params={"date_column": ("dp", "column"), "metric": ("g", "metric"), "title": ("r", "title")},
        output="r",
    ),
    BlockDef(
        name="segment_summary", graph=_segment_summary(), library=_LIB,
        description="count/sum/mean of a metric per segment",
        input_ports={"table": ("g", "table")},
        params={"segment": ("g", "keys"), "metric": ("g", "metric"), "title": ("r", "title")},
        output="r",
    ),
    BlockDef(
        name="correlation_report", graph=_correlation_report(), library=_LIB,
        description="Pearson correlation matrix over the numeric columns, as a report",
        input_ports={"table": ("c", "table")}, params={"title": ("r", "title")}, output="r",
    ),
    BlockDef(
        name="frequency_report", graph=_frequency_report(), library=_LIB,
        description="value counts of a column (most frequent first), as a report",
        input_ports={"table": ("v", "table")},
        params={"column": ("v", "column"), "title": ("r", "title")}, output="r",
    ),
]


def register() -> None:
    """Register the analysis blocks (idempotent). Operators register on import above."""
    for block in _BLOCKS:
        if not is_block(block.name):
            register_block(block)


register()
