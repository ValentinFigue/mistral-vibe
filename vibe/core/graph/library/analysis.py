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

import csv
from datetime import date, datetime
import importlib.resources
from io import StringIO
from pathlib import Path
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


def _cols(value: list[str] | str | None) -> list[str]:
    """Accept a single column name, a list, or None — a bare string is one column, not chars."""
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


@operator(library=_LIB)
async def filter_rows(table: Table, column: str, value: str, op: str = "==") -> Table:
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
async def sort_rows(table: Table, by: str, descending: bool = False) -> Table:
    """Sort rows by a column (None always sorts last)."""
    _need(table, by)
    return _table(table.columns, _sorted_non_null_first(table.rows, by, descending))


@operator(library=_LIB)
async def limit(table: Table, n: int) -> Table:
    """Keep the first ``n`` rows."""
    return _table(table.columns, table.rows[: max(0, n)])


@operator(library=_LIB)
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
async def join(left: Table, right: Table, on: str, how: str = "inner") -> Table:
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


@operator(library=_LIB)
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


@operator(library=_LIB)
async def group_by(table: Table, keys: list[str], metric: str, aggs: list[str] | None = None) -> Table:
    """Group by ``keys`` (empty → overall total) and aggregate ``metric``. aggs ⊆ sum, mean, min,
    max, count (default ["sum"]). Output columns: keys + one per agg (``count`` is a row count).

    Grouping is done with pandas; the output schema (``count`` int, ``{metric}_{agg}`` rounded to
    4) is a stable contract the analysis blocks rely on.
    """
    import pandas as pd

    keys = _cols(keys)
    aggs = _cols(aggs) or ["sum"]
    _need(table, *keys)
    unknown = [a for a in aggs if a not in {*_NUMERIC_AGGS, "count"}]
    if unknown:
        raise ValueError(f"group_by: unknown aggs {unknown}; use sum/mean/min/max/count")
    numeric = [a for a in aggs if a != "count"]
    if numeric:
        _require_numeric(table, metric, "group_by")

    df = _to_df(table)

    def agg_row(sub: Any) -> dict[str, Any]:
        row: dict[str, Any] = {}
        if "count" in aggs:
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

    out_cols = [*keys, *(("count",) if "count" in aggs else ()), *(f"{metric}_{a}" for a in numeric)]
    return _from_df(pd.DataFrame(out_rows, columns=out_cols))


@operator(library=_LIB)
async def describe(table: Table, columns: list[str] | None = None) -> Table:
    """Summary stats per numeric column (empty → all numeric): count, mean, std, min, p25,
    median, p75, max.
    """
    import pandas as pd

    cols = _cols(columns) or [c for c in table.columns if _column_is_numeric(table, c)]
    _need(table, *cols)
    df = _to_df(table)
    out_cols = ["column", "count", "mean", "std", "min", "p25", "median", "p75", "max"]
    rows: list[dict[str, Any]] = []
    for c in cols:
        _require_numeric(table, c, "describe")
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        has = len(s) > 0
        rows.append({
            "column": c,
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
async def top_n(table: Table, by: str, n: int = 10) -> Table:
    """The top ``n`` rows by ``by`` (descending; None-valued rows never count as top)."""
    _need(table, by)
    ranked = _sorted_non_null_first(table.rows, by, descending=True)
    return _table(table.columns, ranked[: max(0, n)])


# --- reshape / window / stats (pandas-backed) ----------------------------------------------


@operator(library=_LIB)
async def pivot(table: Table, index: str, columns: str, values: str, aggfunc: str = "sum") -> Table:
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


@operator(library=_LIB)
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


@operator(library=_LIB)
async def concat(top: Table, bottom: Table) -> Table:
    """Stack two tables' rows (union). Columns are the union; missing cells become null."""
    import pandas as pd

    return _from_df(pd.concat([_to_df(top), _to_df(bottom)], ignore_index=True))


@operator(library=_LIB)
async def fill_missing(table: Table, column: str, method: str = "value", value: Any = None) -> Table:
    """Fill missing values in ``column``. method ∈ value, mean, median, ffill, bfill."""
    import pandas as pd

    _need(table, column)
    if method not in {"value", "mean", "median", "ffill", "bfill"}:
        raise ValueError(f"fill_missing: method must be value/mean/median/ffill/bfill, got {method!r}")
    if method == "value" and value is None:
        raise ValueError("fill_missing: method='value' needs a `value` (or use mean/median/ffill/bfill)")
    df = _to_df(table)
    if method == "value":
        df[column] = df[column].fillna(value)
    elif method in {"mean", "median"}:
        num = pd.to_numeric(df[column], errors="coerce")
        df[column] = num.fillna(getattr(num, method)())
    else:
        df[column] = df[column].ffill() if method == "ffill" else df[column].bfill()
    return _from_df(df)


@operator(library=_LIB)
async def rank(table: Table, by: str, name: str = "rank", descending: bool = True, method: str = "dense") -> Table:
    """Add a ``name`` column ranking rows by ``by`` (1 = top). method ∈ dense, min, first, average."""
    import pandas as pd

    _require_numeric(table, by, "rank")
    if method not in {"dense", "min", "first", "average"}:
        raise ValueError(f"rank: method must be dense/min/first/average, got {method!r}")
    df = _to_df(table)
    ranks = pd.to_numeric(df[by], errors="coerce").rank(ascending=not descending, method=method)
    df[name] = ranks.astype("Int64")
    return _from_df(df)


@operator(name="bin", library=_LIB)
async def bin_column(table: Table, column: str, bins: int = 4, name: str | None = None) -> Table:
    """Bucket a numeric ``column`` into ``bins`` equal-width bins; adds a label column."""
    import pandas as pd

    _require_numeric(table, column, "bin")
    df = _to_df(table)
    out = name or f"{column}_bin"
    df[out] = pd.cut(pd.to_numeric(df[column], errors="coerce"), bins=bins).astype("str")
    return _from_df(df)


@operator(library=_LIB)
async def correlation(table: Table, columns: list[str] | None = None) -> Table:
    """Pearson correlation matrix over numeric columns (a ``column`` label col + one col each)."""
    import pandas as pd

    cols = _cols(columns) or [c for c in table.columns if _column_is_numeric(table, c)]
    _need(table, *cols)
    for c in cols:
        _require_numeric(table, c, "correlation")
    num = _to_df(table)[cols].apply(pd.to_numeric, errors="coerce")
    return _from_df(num.corr().round(4).reset_index(names="column"))


@operator(library=_LIB)
async def pct_change(table: Table, column: str, name: str | None = None) -> Table:
    """Row-over-row percent change of a numeric ``column`` (adds ``{column}_pct_change``)."""
    import pandas as pd

    _require_numeric(table, column, "pct_change")
    df = _to_df(table)
    out = name or f"{column}_pct_change"
    df[out] = (pd.to_numeric(df[column], errors="coerce").pct_change() * 100).round(4)
    return _from_df(df)


@operator(library=_LIB)
async def rolling(table: Table, column: str, window: int, name: str | None = None, stat: str = "mean") -> Table:
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


@operator(library=_LIB)
async def quantile(table: Table, column: str, q: float) -> Table:
    """The ``q``-quantile (``q`` in [0, 1]) of a numeric ``column`` — a 1×1 table (``quantile``)."""
    import pandas as pd

    _require_numeric(table, column, "quantile")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"quantile: q must be in [0, 1], got {q}")
    s = pd.to_numeric(_to_df(table)[column], errors="coerce").dropna()
    val = s.quantile(q) if len(s) else float("nan")
    return _from_df(pd.DataFrame([{"quantile": val}]).round(4))


@operator(library=_LIB)
async def outliers_iqr(table: Table, column: str, k: float = 1.5) -> Table:
    """IQR outliers of a numeric ``column``: values outside [Q1 − k·IQR, Q3 + k·IQR].

    Returns a 1-row summary — ``lower``, ``upper`` (the fences), ``count`` (outliers), ``total``.
    """
    import pandas as pd

    _require_numeric(table, column, "outliers_iqr")
    s = pd.to_numeric(_to_df(table)[column], errors="coerce").dropna()
    if s.empty:
        return _from_df(pd.DataFrame([{"lower": None, "upper": None, "count": 0, "total": 0}]))
    q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    count = int(((s < lower) | (s > upper)).sum())
    row = {"lower": round(lower, 4), "upper": round(upper, 4), "count": count, "total": int(len(s))}
    return _from_df(pd.DataFrame([row]))


@operator(library=_LIB)
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


# --- machine learning (scikit-learn — optional `[ml]` extra) --------------------------------
# sklearn is NOT a base dependency; these ops import it lazily and raise an actionable error when
# it is absent. Results are deterministic for a fixed ``seed`` but depend on the scikit-learn
# version, which the recipe fingerprint does NOT capture — bump CACHE_VERSION when upgrading it.


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # only hit when the [ml] extra isn't installed
        raise ValueError(
            "ML operators need scikit-learn — install the optional extra: "
            "`uv sync --extra ml` (or `pip install 'mistral-vibe[ml]'`)."
        ) from exc


def _ml_xy(table: Table, target: str, features: list[str]) -> tuple[Any, Any]:
    """Build (X, y): one-hot-encode categorical features, drop rows with any missing value."""
    import pandas as pd

    feats = _cols(features)
    if not feats:
        raise ValueError("ml: `features` must list at least one column")
    _need(table, target, *feats)
    df = _to_df(table)[[*feats, target]].dropna()
    if len(df) <= 1:
        raise ValueError("ml: need at least 2 complete rows after dropping missing values")
    return pd.get_dummies(df[feats]), df[target]


def _score_table(value: float) -> Table:
    import pandas as pd

    return _from_df(pd.DataFrame([{"score": round(float(value), 4)}]))


@operator(library=_LIB)
async def ml_regression(
    table: Table,
    target: str,
    features: list[str],
    model: str = "linear",
    metric: str = "r2",
    test_size: float = 0.2,
    seed: int = 0,
) -> Table:
    """Fit a regression model on a train split, score it on the held-out test split — a 1×1 table
    (``score``). model ∈ linear, ridge, tree, rf; metric ∈ r2, rmse, mae. Categorical features are
    one-hot encoded and rows with missing values dropped. Deterministic for a given ``seed``.
    """
    _require_sklearn()
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor

    models = {
        "linear": lambda: LinearRegression(),
        "ridge": lambda: Ridge(random_state=seed),
        "tree": lambda: DecisionTreeRegressor(random_state=seed),
        "rf": lambda: RandomForestRegressor(random_state=seed),
    }
    if model not in models:
        raise ValueError(f"ml_regression: model must be one of {sorted(models)}, got {model!r}")
    if metric not in {"r2", "rmse", "mae"}:
        raise ValueError(f"ml_regression: metric must be r2/rmse/mae, got {metric!r}")
    x, y = _ml_xy(table, target, features)
    y = pd.to_numeric(y, errors="coerce")
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=test_size, random_state=seed)
    est = models[model]()
    est.fit(x_tr, y_tr)
    pred = est.predict(x_te)
    score = {
        "r2": lambda: r2_score(y_te, pred),
        "rmse": lambda: mean_squared_error(y_te, pred) ** 0.5,
        "mae": lambda: mean_absolute_error(y_te, pred),
    }[metric]()
    return _score_table(score)


@operator(library=_LIB)
async def ml_classification(
    table: Table,
    target: str,
    features: list[str],
    model: str = "logreg",
    metric: str = "accuracy",
    test_size: float = 0.2,
    seed: int = 0,
) -> Table:
    """Fit a classifier on a train split, score it on the held-out test split — a 1×1 table
    (``score``). model ∈ logreg, tree, rf; metric ∈ accuracy, f1 (weighted). Categorical features
    are one-hot encoded and rows with missing values dropped. Deterministic for a given ``seed``.
    """
    _require_sklearn()
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    models = {
        "logreg": lambda: LogisticRegression(max_iter=1000, random_state=seed),
        "tree": lambda: DecisionTreeClassifier(random_state=seed),
        "rf": lambda: RandomForestClassifier(random_state=seed),
    }
    if model not in models:
        raise ValueError(f"ml_classification: model must be one of {sorted(models)}, got {model!r}")
    if metric not in {"accuracy", "f1"}:
        raise ValueError(f"ml_classification: metric must be accuracy/f1, got {metric!r}")
    x, y = _ml_xy(table, target, features)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=test_size, random_state=seed)
    est = models[model]()
    est.fit(x_tr, y_tr)
    pred = est.predict(x_te)
    score = (
        accuracy_score(y_te, pred)
        if metric == "accuracy"
        else f1_score(y_te, pred, average="weighted")
    )
    return _score_table(score)


@operator(library=_LIB)
async def ml_cluster(
    table: Table, features: list[str], k: int, metric: str = "silhouette", seed: int = 0
) -> Table:
    """K-means over ``features`` (categoricals one-hot encoded, missing rows dropped) — a 1×1 table
    (``score``): ``silhouette`` (cohesion/separation, higher is better) or ``inertia``.
    Deterministic for a given ``seed``.
    """
    _require_sklearn()
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if metric not in {"silhouette", "inertia"}:
        raise ValueError(f"ml_cluster: metric must be silhouette/inertia, got {metric!r}")
    feats = _cols(features)
    _need(table, *feats)
    x = pd.get_dummies(_to_df(table)[feats].dropna())
    if len(x) <= k:
        raise ValueError(f"ml_cluster: need more rows ({len(x)}) than clusters (k={k})")
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(x)
    score = silhouette_score(x, km.labels_) if metric == "silhouette" else km.inertia_
    return _score_table(score)


# --- data-quality gates --------------------------------------------------------------------


@operator(library=_LIB)
async def expect_columns(table: Table, columns: list[str]) -> Table:
    """Assert the table has the named columns; pass it through unchanged, else fail with a
    clear error the agent can act on.
    """
    missing = [c for c in _cols(columns) if c not in table.columns]
    if missing:
        raise ValueError(f"expect_columns: missing {missing}; available columns are {table.columns}")
    return table


@operator(library=_LIB)
async def expect_no_nulls(table: Table, columns: list[str] | None = None) -> Table:
    """Assert no missing (None/blank) values in ``columns`` (or all); pass through unchanged."""
    check = _cols(columns) or table.columns
    _need(table, *check)
    for c in check:
        bad = sum(1 for r in table.rows if r[c] is None or r[c] == "")
        if bad:
            raise ValueError(f"expect_no_nulls: column {c!r} has {bad} missing value(s)")
    return table


@operator(library=_LIB)
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


@operator(library=_LIB)
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


async def _save_chart(kind: str, table: Table, x: str, y: str, path: str, title: str) -> ChartResult:
    import matplotlib

    matplotlib.use("Agg")  # headless, deterministic; no display backend
    import matplotlib.pyplot as plt
    import pandas as pd

    _need(table, x, y)
    df = _to_df(table)
    ys = pd.to_numeric(df[y], errors="coerce")
    xs = df[x].astype("str")
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    try:
        (ax.bar if kind == "bar" else ax.plot)(xs, ys)
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        fig.autofmt_xdate()
        fig.tight_layout()
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p)
    finally:
        plt.close(fig)
    return ChartResult(path=str(p), kind=kind)


@operator(library=_LIB)
async def bar_chart(table: Table, x: str, y: str, path: str, title: str = "Chart") -> ChartResult:
    """Render a bar chart (``x`` categories, numeric ``y``) to a PNG at ``path``; returns a handle."""
    return await _save_chart("bar", table, x, y, path, title)


@operator(library=_LIB)
async def line_chart(table: Table, x: str, y: str, path: str, title: str = "Chart") -> ChartResult:
    """Render a line chart (``x`` vs numeric ``y``) to a PNG at ``path``; returns a handle."""
    return await _save_chart("line", table, x, y, path, title)


# --- report --------------------------------------------------------------------------------


@operator(library=_LIB)
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


@operator(library=_LIB)
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
