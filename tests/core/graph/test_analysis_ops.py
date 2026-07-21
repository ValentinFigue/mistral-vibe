from __future__ import annotations

import json

import pytest

from vibe.core.graph.cache import CacheStore
from vibe.core.graph.executor import execute
import vibe.core.graph.library.analysis as A
from vibe.core.graph.model import Graph, Node


async def _sales() -> A.Table:
    return await A.sample_dataset(name="sales")


@pytest.mark.asyncio
async def test_sample_dataset_types_inferred() -> None:
    t = await _sales()
    assert {"date", "region", "country", "revenue", "units"} <= set(t.columns)
    # numeric columns are ints/floats, not strings
    assert all(isinstance(r["revenue"], (int, float)) for r in t.rows)
    assert all(isinstance(r["units"], int) for r in t.rows)
    assert all(isinstance(r["region"], str) for r in t.rows)


@pytest.mark.asyncio
async def test_group_by_sum_count_and_overall_total() -> None:
    t = await _sales()
    g = await A.group_by(t, keys=["country"], metric="revenue", aggs=["sum", "count"])
    assert set(g.columns) == {"country", "count", "revenue_sum"}
    assert len(g.rows) == len({r["country"] for r in t.rows})
    # keys=[] → single overall total row
    total = await A.group_by(t, keys=[], metric="revenue", aggs=["sum"])
    assert len(total.rows) == 1
    assert total.rows[0]["revenue_sum"] == pytest.approx(sum(r["revenue"] for r in t.rows))


@pytest.mark.asyncio
async def test_group_by_non_numeric_metric_errors() -> None:
    t = await _sales()
    with pytest.raises(ValueError, match="not numeric"):
        await A.group_by(t, keys=["country"], metric="region", aggs=["sum"])


@pytest.mark.asyncio
async def test_missing_column_lists_available() -> None:
    t = await _sales()
    with pytest.raises(ValueError, match="not found; available columns are"):
        await A.filter_rows(t, column="nope", op="==", value="x")


@pytest.mark.asyncio
async def test_filter_rows_numeric_coercion() -> None:
    t = await _sales()
    out = await A.filter_rows(t, column="revenue", op=">", value="3000")
    assert out.rows and all(r["revenue"] > 3000 for r in out.rows)


@pytest.mark.asyncio
async def test_derive_column_and_cast() -> None:
    t = await _sales()
    out = await A.derive_column(t, name="margin", left="revenue", op="-", right="cost")
    assert "margin" in out.columns
    assert out.rows[0]["margin"] == pytest.approx(t.rows[0]["revenue"] - t.rows[0]["cost"])
    # ratio with a constant
    ratio = await A.derive_column(t, name="half", left="revenue", op="/", right="2")
    assert ratio.rows[0]["half"] == pytest.approx(t.rows[0]["revenue"] / 2)


@pytest.mark.asyncio
async def test_date_part_and_describe() -> None:
    t = await _sales()
    dp = await A.date_part(t, column="date", part="month")
    assert "date_month" in dp.columns and dp.rows[0]["date_month"] in range(1, 13)
    desc = await A.describe(t, columns=["revenue"])
    row = desc.rows[0]
    assert row["column"] == "revenue" and row["count"] == len(t.rows)


@pytest.mark.asyncio
async def test_join_and_value_counts() -> None:
    sales = await _sales()
    customers = await A.sample_dataset(name="customers")
    joined = await A.join(sales, customers, on="country", how="left")
    assert "plan" in joined.columns and len(joined.rows) == len(sales.rows)
    vc = await A.value_counts(sales, column="region")
    assert set(vc.columns) == {"value", "count"}
    assert vc.rows[0]["count"] >= vc.rows[-1]["count"]  # sorted desc


@pytest.mark.asyncio
async def test_to_markdown_renders_table() -> None:
    t = await A.top_n(await _sales(), by="revenue", n=2)
    report = await A.to_markdown(t, title="Top", max_rows=10)
    assert report.markdown.startswith("# Top")
    assert "| revenue |" in report.markdown


@pytest.mark.asyncio
async def test_read_csv_parses(tmp_path) -> None:
    p = tmp_path / "d.csv"
    p.write_text("a,b\n1,x\n2,y\n")
    t = await A.read_csv(path=str(p), content_fp="unused")
    assert t.columns == ["a", "b"]
    assert t.rows == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]


@pytest.mark.asyncio
async def test_pipeline_incremental(cache: CacheStore) -> None:
    def build(n: int) -> Graph:
        g = Graph()
        g.add(Node(id="src", op="sample_dataset", params={"name": "sales"}))
        g.add(Node(id="grp", op="group_by",
                   params={"keys": ["country"], "metric": "revenue", "aggs": ["sum"]},
                   inputs={"table": "src"}))
        g.add(Node(id="top", op="top_n", params={"by": "revenue_sum", "n": n}, inputs={"table": "grp"}))
        g.add(Node(id="rep", op="to_markdown", params={"title": "T", "max_rows": 50}, inputs={"table": "top"}))
        return g

    _, cold = await execute(build(5), cache)
    assert set(cold.fresh()) == {"src", "grp", "top", "rep"}
    # Edit only top_n's n → src + grp cached; top + rep recompute.
    _, rerun = await execute(build(3), cache)
    assert set(rerun.cached()) == {"src", "grp"}
    assert set(rerun.fresh()) == {"top", "rep"}


@pytest.mark.asyncio
async def test_analysis_blocks_expand_and_run(cache: CacheStore) -> None:
    g = Graph()
    g.add(Node(id="src", op="sample_dataset", params={"name": "sales"}))
    g.add(Node(id="prof", op="quick_profile", params={"title": "Profile"}, inputs={"table": "src"}))
    values, report = await execute(g, cache)
    assert "prof/r" in values  # block expanded to its inner report node
    md = json.loads(cache.get(values["prof/r"].fingerprint).decode())["markdown"]
    assert "# Profile" in md


@pytest.mark.asyncio
async def test_optional_params_may_be_omitted(cache: CacheStore) -> None:
    # O1: to_markdown omits title+max_rows; group_by omits aggs → defaults fill in, no error.
    g = Graph()
    g.add(Node(id="src", op="sample_dataset", params={"name": "sales"}))
    g.add(Node(id="grp", op="group_by", params={"keys": ["country"], "metric": "revenue"},
               inputs={"table": "src"}))
    g.add(Node(id="rep", op="to_markdown", inputs={"table": "grp"}))
    values, report = await execute(g, cache)
    assert set(report.fresh()) == {"src", "grp", "rep"}
    md = json.loads(cache.get(values["rep"].fingerprint).decode())["markdown"]
    assert md.startswith("# Report")  # default title
    assert "revenue_sum" in md  # default agg = sum


@pytest.mark.asyncio
async def test_type_mismatch_rejected_at_validate(cache: CacheStore) -> None:
    # O2: wiring a Report into a table input is caught before execution, with a clear message.
    from vibe.core.graph.executor import GraphValidationError

    g = Graph()
    g.add(Node(id="src", op="sample_dataset", params={"name": "sales"}))
    g.add(Node(id="rep", op="to_markdown", inputs={"table": "src"}))
    g.add(Node(id="bad", op="group_by", params={"keys": ["country"], "metric": "revenue"},
               inputs={"table": "rep"}))
    with pytest.raises(GraphValidationError, match="expects Table but node 'rep' produces Report"):
        await execute(g, cache)


@pytest.mark.asyncio
async def test_shape_operators() -> None:
    t = await _sales()
    assert (await A.select_columns(t, columns=["country", "revenue"])).columns == ["country", "revenue"]
    renamed = await A.rename_columns(t, mapping={"revenue": "rev"})
    assert "rev" in renamed.columns and "revenue" not in renamed.columns
    casted = await A.cast_column(t, column="units", type="float")
    assert all(isinstance(r["units"], float) for r in casted.rows)
    assert len((await A.limit(t, n=3)).rows) == 3
    assert len((await A.distinct(t, columns=["region"])).rows) == len({r["region"] for r in t.rows})
    asc = await A.sort_rows(t, by="revenue", descending=False)
    revs = [r["revenue"] for r in asc.rows]
    assert revs == sorted(revs)
    # drop_missing removes blank rows
    holed = A.Table(columns=["a", "b"], rows=[{"a": 1, "b": "x"}, {"a": None, "b": "y"}])
    assert len((await A.drop_missing(holed, columns=["a"])).rows) == 1


@pytest.mark.asyncio
async def test_scalar_column_params_are_accepted() -> None:
    # A bare string where a list is expected must mean one column, not per-character.
    t = await _sales()
    g = await A.group_by(t, keys="country", metric="revenue", aggs=["sum"])  # type: ignore[arg-type]
    assert "country" in g.columns and len(g.rows) == len({r["country"] for r in t.rows})
    assert (await A.select_columns(t, columns="region")).columns == ["region"]  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_top_n_puts_nulls_last() -> None:
    # Regression (temper #1): a None metric must never rank as "top".
    t = A.Table(columns=["k", "v"], rows=[{"k": "a", "v": 10}, {"k": "b", "v": None}, {"k": "c", "v": 30}])
    top2 = await A.top_n(t, by="v", n=2)
    assert [r["k"] for r in top2.rows] == ["c", "a"]


@pytest.mark.asyncio
async def test_read_csv_keeps_zero_padded_ids(tmp_path) -> None:
    # Regression (temper #2): zero-padded identifiers must stay strings, not become ints.
    p = tmp_path / "z.csv"
    p.write_text("zip,n\n01234,1\n00077,2\n")
    t = await A.read_csv(path=str(p), content_fp="x")
    assert [r["zip"] for r in t.rows] == ["01234", "00077"]
    assert [r["n"] for r in t.rows] == [1, 2]  # ordinary ints still inferred


@pytest.mark.asyncio
async def test_join_rejects_duplicate_right_keys() -> None:
    # Regression (temper #3): a non-unique right table would silently drop matches.
    left = A.Table(columns=["country", "x"], rows=[{"country": "fr", "x": 1}])
    dup = A.Table(columns=["country", "y"], rows=[{"country": "fr", "y": 1}, {"country": "fr", "y": 2}])
    with pytest.raises(ValueError, match="duplicate 'country' values"):
        await A.join(left, dup, on="country", how="left")


@pytest.mark.asyncio
async def test_all_analysis_blocks_run(cache: CacheStore) -> None:
    def wrap(block_id: str, params: dict) -> Graph:
        g = Graph()
        g.add(Node(id="src", op="sample_dataset", params={"name": "sales"}))
        g.add(Node(id="b", op=block_id, params=params, inputs={"table": "src"}))
        return g

    for block_id, params in (
        ("rank_by", {"group_key": "country", "metric": "revenue",
                     "rank_by_column": "revenue_sum", "n": 3, "title": "Rank"}),
        ("trend_by_period", {"date_column": "date", "period": "month", "group_key": "date_month",
                             "metric": "revenue", "sort_key": "date_month", "title": "Trend"}),
        ("segment_summary", {"segment": "region", "metric": "revenue", "title": "Segments"}),
    ):
        values, _ = await execute(wrap(block_id, params), cache)
        md = json.loads(cache.get(values["b/r"].fingerprint).decode())["markdown"]
        assert md.startswith("# ")
