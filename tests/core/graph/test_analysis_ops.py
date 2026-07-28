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
    # a comparison yields a 1/0 binary column (e.g. a target threshold)
    hi = await A.derive_column(t, name="hi", left="revenue", op=">", right="2000")
    assert all(r["hi"] == (1 if r["revenue"] > 2000 else 0) for r in hi.rows)
    assert set(r["hi"] for r in hi.rows) <= {0, 1}


@pytest.mark.asyncio
async def test_date_part_and_describe() -> None:
    t = await _sales()
    dp = await A.date_part(t, column="date", part="month")
    assert "date_month" in dp.columns and dp.rows[0]["date_month"] in range(1, 13)
    desc = await A.describe(t, columns=["revenue"])
    row = desc.rows[0]
    # the label column is "field" (not the SQL reserved word "column")
    assert row["field"] == "revenue" and row["count"] == len(t.rows)
    assert "column" not in desc.columns
    # median/quartiles are now part of describe, ordered min ≤ p25 ≤ median ≤ p75 ≤ max
    assert set(desc.columns) >= {"min", "p25", "median", "p75", "max"}
    assert row["min"] <= row["p25"] <= row["median"] <= row["p75"] <= row["max"]


@pytest.mark.asyncio
async def test_quantile_outliers_and_distribution() -> None:
    t = await _sales()
    q90 = await A.quantile(t, column="revenue", q=0.9)
    assert q90.columns == ["quantile"] and len(q90.rows) == 1
    assert isinstance(q90.rows[0]["quantile"], (int, float))

    out = await A.outliers(t, column="revenue")  # method="iqr" default
    r = out.rows[0]
    assert set(out.columns) == {"lower", "upper", "count", "total"}
    assert r["lower"] <= r["upper"] and 0 <= r["count"] <= r["total"] == len(t.rows)

    dist = await A.distribution(t, column="revenue")
    assert set(dist.columns) == {"mean", "std", "skewness", "kurtosis"} and len(dist.rows) == 1

    # q out of range is a clear error
    with pytest.raises(ValueError, match="q must be in"):
        await A.quantile(t, column="revenue", q=1.5)


@pytest.mark.asyncio
async def test_answer_single_value_and_rounding() -> None:
    # A 1×1 table → the report text is exactly the value; decimals rounds numerics.
    r = await A.answer(A.Table(columns=["quantile"], rows=[{"quantile": 42.17346}]), decimals=2)
    assert r.markdown == "42.17"
    # non-numeric passes through untouched even with decimals set
    r2 = await A.answer(A.Table(columns=["top"], rows=[{"top": "emea"}]), decimals=2)
    assert r2.markdown == "emea"
    # not a 1×1 table → clear error
    with pytest.raises(ValueError, match="expected a 1.1 table"):
        await A.answer(A.Table(columns=["a", "b"], rows=[{"a": 1, "b": 2}]))


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
async def test_read_csv_recognizes_standard_na_tokens(tmp_path) -> None:
    # Regression: only '' used to be treated as missing; standard na_values tokens (matching
    # pandas.read_csv's defaults) must also become None, in both string and numeric columns.
    p = tmp_path / "na.csv"
    p.write_text("dept,age\neng,30\nN/A,NaN\nnull,25\nsales,None\n")
    t = await A.read_csv(path=str(p), content_fp="x")
    assert [r["dept"] for r in t.rows] == ["eng", None, None, "sales"]
    assert [r["age"] for r in t.rows] == [30, None, 25, None]


@pytest.mark.asyncio
async def test_read_csv_na_token_does_not_downgrade_numeric_column_to_strings(tmp_path) -> None:
    # Regression: a stray NA token used to make _all(int)/_all(float) raise, silently
    # stringifying the *whole* column, including the genuinely-numeric values.
    p = tmp_path / "na2.csv"
    p.write_text("n\n1\nNA\n3\n")
    t = await A.read_csv(path=str(p), content_fp="x")
    assert t.rows == [{"n": 1}, {"n": None}, {"n": 3}]
    assert all(isinstance(r["n"], int) for r in t.rows if r["n"] is not None)


@pytest.mark.asyncio
async def test_sql_is_null_groupby_matches_pandas_na_handling(tmp_path) -> None:
    # Regression: the exact DABench failure mode — "mean of X grouped by whether Y is
    # null" came out wrong because a handful of "N/A"-token rows were misclassified as
    # non-null by DuckDB's IS NULL, before the _coerce_column fix. Cross-check against
    # plain pandas.read_csv's default na_values handling of the same file.
    p = tmp_path / "grp.csv"
    p.write_text("tree,nsnps\nA,10\n,20\nN/A,30\nB,40\nnull,50\n")
    t = await A.read_csv(path=str(p), content_fp="x")
    out = await A.sql(
        query="""
        SELECT tree IS NULL AS is_null, AVG(nsnps) AS mean_nsnps
        FROM t1 GROUP BY 1 ORDER BY 1
        """,
        t1=t,
    )
    got = {r["is_null"]: r["mean_nsnps"] for r in out.rows}

    import pandas as pd

    ref = pd.read_csv(p)
    is_null = ref["tree"].isna()
    expected_true = float(ref.loc[is_null, "nsnps"].mean())
    expected_false = float(ref.loc[~is_null, "nsnps"].mean())
    assert got[True] == pytest.approx(expected_true)
    assert got[False] == pytest.approx(expected_false)


@pytest.mark.asyncio
async def test_filter_rows_is_null_and_is_not_null() -> None:
    t = A.Table(
        columns=["dept", "salary"],
        rows=[
            {"dept": "eng", "salary": 100},
            {"dept": None, "salary": None},
            {"dept": "sales", "salary": 90},
        ],
    )
    null_rows = await A.filter_rows(t, column="dept", op="is_null")
    assert [r["dept"] for r in null_rows.rows] == [None]
    not_null_rows = await A.filter_rows(t, column="salary", op="is_not_null")
    assert [r["salary"] for r in not_null_rows.rows] == [100, 90]
    # numeric column: is_null must not raise ("'val' is numeric but value '' isn't").
    numeric_null = await A.filter_rows(t, column="salary", op="is_null")
    assert len(numeric_null.rows) == 1


@pytest.mark.asyncio
async def test_filter_rows_is_null_value_is_optional() -> None:
    t = A.Table(columns=["a"], rows=[{"a": 1}, {"a": None}])
    out = await A.filter_rows(t, column="a", op="is_null")  # no `value` kwarg
    assert len(out.rows) == 1


@pytest.mark.asyncio
async def test_join_is_a_real_merge_multiplying_dup_keys() -> None:
    # pandas merge: duplicate keys on the right multiply matching rows (standard SQL semantics),
    # rather than the old lookup that silently dropped matches.
    left = A.Table(columns=["country", "x"], rows=[{"country": "fr", "x": 1}])
    dup = A.Table(columns=["country", "y"], rows=[{"country": "fr", "y": 1}, {"country": "fr", "y": 2}])
    out = await A.join(left, dup, on="country", how="left")
    assert out.columns == ["country", "x", "y"]
    assert sorted(r["y"] for r in out.rows) == [1, 2]
    assert all(r["x"] == 1 for r in out.rows)  # left row duplicated across both matches


def test_importing_analysis_does_not_load_pandas() -> None:
    # crit #1: graph_patch imports this module at tool discovery (every startup); pandas must be
    # imported lazily inside op bodies, not at module load, so the CLI startup stays lean.
    import subprocess
    import sys

    code = (
        "import sys; import vibe.core.graph.library.analysis;"
        " assert 'pandas' not in sys.modules, 'pandas loaded at import time';"
        " assert 'matplotlib' not in sys.modules, 'matplotlib loaded at import time';"
        " assert 'duckdb' not in sys.modules, 'duckdb loaded at import time';"
        " assert 'sklearn' not in sys.modules, 'sklearn loaded at import time'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_sql_operator_query_join_and_sandbox() -> None:
    t = await _sales()
    customers = await A.sample_dataset(name="customers")
    # single-table aggregation
    r = await A.sql(query="SELECT country, sum(revenue) AS rev FROM t1 GROUP BY country ORDER BY rev DESC LIMIT 3", t1=t)
    assert r.columns == ["country", "rev"] and len(r.rows) == 3
    assert r.rows[0]["rev"] >= r.rows[-1]["rev"]  # ordered
    # join across two wired tables
    j = await A.sql(
        query="SELECT s.country, c.plan, sum(s.revenue) rev FROM t1 s JOIN t2 c ON s.country=c.country GROUP BY 1,2",
        t1=t, t2=customers,
    )
    assert {"country", "plan", "rev"} == set(j.columns)
    # sandbox: no filesystem access
    with pytest.raises(ValueError, match="sql: query failed"):
        await A.sql(query="SELECT * FROM read_csv('/etc/passwd')", t1=t)


@pytest.mark.asyncio
async def test_sql_operator_is_pure_through_executor(cache: CacheStore) -> None:
    # The sql op must round-trip losslessly so verify_purity (cache soundness) holds.
    g = Graph()
    g.add(Node(id="src", op="sample_dataset", params={"name": "sales"}))
    g.add(Node(id="q", op="sql",
               params={"query": "SELECT country, sum(revenue) AS rev FROM t1 GROUP BY country ORDER BY country"},
               inputs={"t1": "src"}))
    await execute(g, cache)
    _, rep = await execute(g, cache, verify_purity=True)
    assert rep.states["q"] == "cached"


@pytest.mark.asyncio
async def test_df_bridge_round_trips_json_native(cache: CacheStore) -> None:
    # The pandas bridge must produce JSON-native, NaN→None values so the cache round-trip +
    # verify_purity equality hold (crit #2).
    t = A.Table(
        columns=["i", "f", "s", "u"],
        rows=[{"i": 1, "f": 1.5, "s": "x", "u": "café"}, {"i": 2, "f": None, "s": "y", "u": "β"}],
    )
    back = A._from_df(A._to_df(t))
    assert back.rows[0]["i"] == 1 and isinstance(back.rows[0]["i"], int)  # int stays int
    assert back.rows[1]["f"] is None  # NaN → None
    assert back.rows[0]["u"] == "café"
    # idempotent (verify_purity compares model_dump across a rebuild)
    assert A._from_df(A._to_df(back)).model_dump() == back.model_dump()


@pytest.mark.asyncio
async def test_new_pandas_ops() -> None:
    t = await _sales()
    piv = await A.pivot(t, index="region", columns="product", values="revenue", aggfunc="sum")
    assert piv.columns[0] == "region" and len(piv.columns) > 1
    corr = await A.correlation(t, columns=["units", "revenue"])
    assert corr.rows[0]["field"] == "units" and corr.rows[0]["units"] == 1.0
    ranked = await A.rank(t, by="revenue", name="rk")
    assert "rk" in ranked.columns and min(r["rk"] for r in ranked.rows) == 1
    binned = await A.bin_column(t, column="revenue", bins=3)
    assert "revenue_bin" in binned.columns
    ml = await A.melt(t, id_vars=["region"], value_vars=["revenue", "cost"])
    assert ml.columns == ["region", "variable", "value"] and len(ml.rows) == 2 * len(t.rows)
    filled = await A.fill_missing(
        A.Table(columns=["x"], rows=[{"x": 1}, {"x": None}, {"x": 3}]), column="x", method="mean"
    )
    assert [r["x"] for r in filled.rows] == [1.0, 2.0, 3.0]
    doubled = await A.concat(t, t)
    assert len(doubled.rows) == 2 * len(t.rows)
    seq = A.Table(columns=["v"], rows=[{"v": 10}, {"v": 20}, {"v": 40}])
    pc = await A.pct_change(seq, column="v")
    assert pc.rows[0]["v_pct_change"] is None and pc.rows[1]["v_pct_change"] == 100.0
    roll = await A.rolling(seq, column="v", window=2, stat="mean")
    assert roll.rows[1]["v_rolling_mean"] == 15.0


@pytest.mark.asyncio
async def test_data_quality_gates() -> None:
    t = await _sales()
    assert (await A.expect_columns(t, columns=["region", "revenue"])).columns == t.columns
    with pytest.raises(ValueError, match="expect_columns: missing"):
        await A.expect_columns(t, columns=["nope"])
    with pytest.raises(ValueError, match="not unique"):
        await A.expect_unique(t, columns=["product"])
    holed = A.Table(columns=["a"], rows=[{"a": 1}, {"a": None}])
    with pytest.raises(ValueError, match="missing value"):
        await A.expect_no_nulls(holed, columns=["a"])


@pytest.mark.asyncio
async def test_sink_ops_write_file_and_return_handle(tmp_path) -> None:
    t = await _sales()
    csv_path = tmp_path / "out.csv"
    exp = await A.to_csv(t, path=str(csv_path))
    assert csv_path.exists() and exp.rows == len(t.rows) and "region" in exp.columns
    ranked = await A.group_by(t, keys=["country"], metric="revenue", aggs=["sum"])
    png = tmp_path / "chart.png"
    chart = await A.chart(ranked, kind="bar", x="country", y="revenue_sum", path=str(png), title="Top")
    assert png.exists() and png.stat().st_size > 0 and chart.kind == "bar"
    # the value stays a tiny handle, not the image bytes
    assert set(chart.model_dump()) == {"path", "kind"}
    line_png = tmp_path / "line.png"
    line = await A.chart(ranked, kind="line", x="country", y="revenue_sum", path=str(line_png), title="Trend")
    assert line_png.exists() and line.kind == "line"


@pytest.mark.asyncio
async def test_all_analysis_blocks_run(cache: CacheStore) -> None:
    def wrap(block_id: str, params: dict) -> Graph:
        g = Graph()
        g.add(Node(id="src", op="sample_dataset", params={"name": "sales"}))
        g.add(Node(id="b", op=block_id, params=params, inputs={"table": "src"}))
        return g

    for block_id, params in (
        # rank_by/trend_by_period no longer take the derived column names — computed via templates
        ("rank_by", {"group_key": "country", "metric": "revenue", "n": 3, "title": "Rank"}),
        ("trend_by_period", {"date_column": "date", "period": "month", "metric": "revenue", "title": "Trend"}),
        ("month_over_month_growth", {"date_column": "date", "metric": "revenue", "title": "MoM"}),
        ("segment_summary", {"segment": "region", "metric": "revenue", "title": "Segments"}),
        ("correlation_report", {"title": "Correlations"}),
        ("frequency_report", {"column": "region", "title": "Regions"}),
    ):
        values, _ = await execute(wrap(block_id, params), cache)
        md = json.loads(cache.get(values["b/r"].fingerprint).decode())["markdown"]
        assert md.startswith("# ")


@pytest.mark.timeout(90)  # cold sklearn/scipy import + fits across the model zoo
@pytest.mark.asyncio
async def test_ml_operators_run_and_are_seeded_deterministic() -> None:
    # scikit-learn is a base dependency, so this always runs. Cheap fits keep it fast; determinism
    # is what matters for cache correctness.
    t = await _sales()

    reg = await A.ml_regression(t, target="revenue", features=["units", "cost"], model="linear", metric="r2")
    assert reg.columns == ["score"] and isinstance(reg.rows[0]["score"], (int, float))
    # every new regression model + the mse metric author cleanly
    for model in ("ridge", "lasso", "tree", "rf", "gbr", "knn", "svr"):
        out = await A.ml_regression(t, target="revenue", features=["units", "cost"], model=model, metric="mse")
        assert out.columns == ["score"]

    clf1 = await A.ml_classification(t, target="region", features=["revenue", "units", "cost"], model="tree")
    clf2 = await A.ml_classification(t, target="region", features=["revenue", "units", "cost"], model="tree")
    assert clf1.rows[0]["score"] == clf2.rows[0]["score"]  # fixed random_state → deterministic
    for model in ("logreg", "rf", "gbm", "knn", "svc", "nb"):
        out = await A.ml_classification(t, target="region", features=["revenue", "units"], model=model, scale=True)
        assert out.columns == ["score"]

    clus = await A.ml_cluster(t, features=["revenue", "units"], k=2)
    assert clus.columns == ["score"] and -1.0 <= clus.rows[0]["score"] <= 1.0  # silhouette range

    with pytest.raises(ValueError, match="model must be one of"):
        await A.ml_regression(t, target="revenue", features=["units"], model="nope")


@pytest.mark.timeout(60)
@pytest.mark.asyncio
async def test_ml_regression_matches_sklearn_reference() -> None:
    # The DABench guarantee: our op reproduces a direct scikit-learn computation for the same setup.
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    t = await _sales()
    df = A._to_df(t)[["units", "cost", "revenue"]].dropna()
    x = pd.get_dummies(df[["units", "cost"]])
    y = pd.to_numeric(df["revenue"], errors="coerce")

    # evaluate="full": fit + score on all rows, MSE
    expected_full = mean_squared_error(y, LinearRegression().fit(x, y).predict(x))
    got_full = await A.ml_regression(
        t, target="revenue", features=["units", "cost"], model="linear", metric="mse", evaluate="full"
    )
    assert got_full.rows[0]["score"] == pytest.approx(expected_full, rel=1e-9)

    # evaluate="holdout" with the exact split the op uses (random_state=42, test_size=0.2), RMSE
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42)
    expected_ho = mean_squared_error(y_te, LinearRegression().fit(x_tr, y_tr).predict(x_te)) ** 0.5
    got_ho = await A.ml_regression(
        t, target="revenue", features=["units", "cost"], model="linear", metric="rmse",
        evaluate="holdout", test_size=0.2, random_state=42,
    )
    assert got_ho.rows[0]["score"] == pytest.approx(expected_ho, rel=1e-9)


def _clf_table() -> A.Table:
    return A.Table(
        columns=["grp", "x", "y"],
        rows=[
            {"grp": "a", "x": 1.0, "y": 0}, {"grp": "a", "x": 1.1, "y": 0}, {"grp": "a", "x": 1.2, "y": 0},
            {"grp": "a", "x": 1.3, "y": 0}, {"grp": "a", "x": 1.4, "y": 0}, {"grp": "a", "x": 1.5, "y": 0},
            {"grp": "a", "x": 1.6, "y": 0}, {"grp": "a", "x": 1.7, "y": 0},
            {"grp": "b", "x": 9.0, "y": 1}, {"grp": "b", "x": 9.5, "y": 1},
        ],
    )


@pytest.mark.timeout(30)
@pytest.mark.asyncio
async def test_ml_classification_class_weight_balanced_matches_sklearn() -> None:
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    t = _clf_table()
    df = A._to_df(t)
    x = pd.get_dummies(df[["grp", "x"]])
    y = df["y"]
    expected = accuracy_score(
        y, LogisticRegression(class_weight="balanced", max_iter=100, random_state=42).fit(x, y).predict(x)
    )
    got = await A.ml_classification(
        t, target="y", features=["grp", "x"], model="logreg", evaluate="full",
        class_weight="balanced", random_state=42,
    )
    assert got.rows[0]["score"] == pytest.approx(expected, rel=1e-9)


@pytest.mark.timeout(30)
@pytest.mark.asyncio
async def test_ml_classification_solver_liblinear_matches_sklearn() -> None:
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    t = _clf_table()
    df = A._to_df(t)
    x = pd.get_dummies(df[["grp", "x"]])
    y = df["y"]
    expected = accuracy_score(y, LogisticRegression(solver="liblinear", max_iter=100, random_state=42).fit(x, y).predict(x))
    got = await A.ml_classification(
        t, target="y", features=["grp", "x"], model="logreg", evaluate="full",
        solver="liblinear", random_state=42,
    )
    assert got.rows[0]["score"] == pytest.approx(expected, rel=1e-9)


@pytest.mark.timeout(30)
@pytest.mark.asyncio
async def test_ml_classification_max_iter_default_matches_sklearn_default() -> None:
    # Regression: max_iter was hardcoded to 1000, contradicting the docstring's claim of
    # reproducing sklearn defaults (sklearn's own default is 100).
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    t = _clf_table()
    df = A._to_df(t)
    x = pd.get_dummies(df[["grp", "x"]])
    y = df["y"]
    expected = accuracy_score(y, LogisticRegression(random_state=42).fit(x, y).predict(x))
    got = await A.ml_classification(t, target="y", features=["grp", "x"], model="logreg", evaluate="full", random_state=42)
    assert got.rows[0]["score"] == pytest.approx(expected, rel=1e-9)


@pytest.mark.asyncio
async def test_ml_classification_class_weight_rejected_for_unsupported_model() -> None:
    t = _clf_table()
    with pytest.raises(ValueError, match="class_weight is not supported for model"):
        await A.ml_classification(t, target="y", features=["x"], model="knn", class_weight="balanced")


@pytest.mark.asyncio
async def test_ml_classification_solver_rejected_for_non_logreg_model() -> None:
    t = _clf_table()
    with pytest.raises(ValueError, match="solver is only supported for model"):
        await A.ml_classification(t, target="y", features=["x"], model="rf", solver="liblinear")


@pytest.mark.timeout(30)
@pytest.mark.asyncio
async def test_ml_regression_drop_first_matches_pandas_get_dummies() -> None:
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    t = await _sales()
    df = A._to_df(t)[["region", "units", "revenue"]].dropna()
    x_drop = pd.get_dummies(df[["region", "units"]], drop_first=True)
    y = pd.to_numeric(df["revenue"], errors="coerce")
    expected = mean_squared_error(y, LinearRegression().fit(x_drop, y).predict(x_drop))
    got = await A.ml_regression(
        t, target="revenue", features=["region", "units"], model="linear", metric="mse",
        evaluate="full", drop_first=True,
    )
    assert got.rows[0]["score"] == pytest.approx(expected, rel=1e-9)
    # sanity: drop_first actually changes the encoding (one fewer dummy column per category)
    x_keep = pd.get_dummies(df[["region", "units"]])
    assert x_drop.shape[1] < x_keep.shape[1]


@pytest.mark.asyncio
async def test_ml_xy_note_surfaces_dropped_row_count() -> None:
    t = A.Table(
        columns=["x", "y"],
        rows=[
            {"x": 1, "y": 1}, {"x": 2, "y": 2}, {"x": None, "y": 3}, {"x": 4, "y": None}, {"x": 5, "y": 5},
        ],
    )
    out = await A.ml_regression(t, target="y", features=["x"], model="linear", evaluate="full")
    assert out.notes and "dropped 2/5" in out.notes[0]
    # answer() still accepts the 1x1 score table unchanged despite the extra `notes` field
    ans = await A.answer(out, decimals=2)
    assert isinstance(ans.markdown, str)


@pytest.mark.asyncio
async def test_ml_xy_no_note_when_nothing_dropped() -> None:
    t = A.Table(columns=["x", "y"], rows=[{"x": 1, "y": 1}, {"x": 2, "y": 2}, {"x": 3, "y": 3}])
    out = await A.ml_regression(t, target="y", features=["x"], model="linear", evaluate="full")
    assert out.notes == []


@pytest.mark.timeout(60)
@pytest.mark.asyncio
async def test_ml_predict_returns_aligned_predictions() -> None:
    t = await _sales()
    complete = len(A._to_df(t)[["units", "cost", "revenue"]].dropna())
    out = await A.ml_predict(
        t, target="revenue", features=["units", "cost"], model="linear", task="regression",
        evaluate="full", on="all",
    )
    assert out.columns == ["units", "cost", "revenue", "revenue_pred"]
    assert len(out.rows) == complete
    assert all(isinstance(r["revenue_pred"], float) for r in out.rows)
    with pytest.raises(ValueError, match="not valid for task"):
        await A.ml_predict(t, target="revenue", features=["units"], model="logreg", task="regression")


def test_enum_defaults_are_within_allowed() -> None:
    # Narrowing guard (crit#1): every analysis enum op's default must be one of its Literal values,
    # so annotating params as Literal never rejects a previously-valid default.
    from vibe.core.graph.operators import registered_operators

    for name, spec in registered_operators().items():
        if spec.library != "analysis":
            continue
        for param, allowed in spec.allowed_values.items():
            if param in spec.defaults and spec.defaults[param] is not None:
                default = spec.defaults[param]
                for v in default if isinstance(default, list) else [default]:
                    assert v in allowed, f"{name}.{param} default {v!r} not in {allowed}"


@pytest.mark.asyncio
async def test_read_csv_and_fingerprint_expand_tilde(tmp_path, monkeypatch) -> None:
    from vibe.core.graph.fingerprint import content_hash

    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "d.csv").write_text("a,b\n1,x\n")
    # The fingerprint autofill path (content_hash) and read_csv resolve ~ to the same file.
    fp = content_hash("~/d.csv")
    assert fp  # no OSError
    t = await A.read_csv(path="~/d.csv", content_fp=fp)
    assert t.columns == ["a", "b"] and t.rows == [{"a": 1, "b": "x"}]


class _StubLLM:
    """A deterministic stand-in for the injected LLMCaller — no backend needed."""

    def __init__(self, *, labels: str | None = None, prose: str = "A concise takeaway.") -> None:
        self._labels, self._prose, self.calls = labels, prose, 0

    async def __call__(self, prompt: str, *, system: str | None = None, max_tokens: int = 1024) -> str:
        self.calls += 1
        return self._labels if "JSON array" in prompt else self._prose


@pytest.mark.asyncio
async def test_narrate_returns_prose_report() -> None:
    t = await _sales()
    llm = _StubLLM(prose="Revenue skews to a few countries.")
    rep = await A.narrate(t, goal="revenue concentration", llm=llm)
    assert isinstance(rep, A.Report) and rep.markdown == "Revenue skews to a few countries."
    assert llm.calls == 1  # one LLM call


@pytest.mark.asyncio
async def test_classify_adds_label_column() -> None:
    t = A.Table(columns=["text"], rows=[{"text": "great"}, {"text": "broken"}, {"text": "ok?"}])
    llm = _StubLLM(labels='["praise", "bug", "question"]')
    out = await A.classify(t, column="text", labels=["praise", "bug", "question"], llm=llm)
    assert out.columns == ["text", "text_label"]
    assert [r["text_label"] for r in out.rows] == ["praise", "bug", "question"]


@pytest.mark.asyncio
async def test_classify_rejects_bad_model_output() -> None:
    t = A.Table(columns=["text"], rows=[{"text": "a"}, {"text": "b"}])
    with pytest.raises(ValueError, match="expected 2 labels"):  # count mismatch
        await A.classify(t, column="text", labels=["x", "y"], llm=_StubLLM(labels='["x"]'))
    with pytest.raises(ValueError, match="outside"):  # label not in the allowed set
        await A.classify(t, column="text", labels=["x", "y"], llm=_StubLLM(labels='["x", "z"]'))


@pytest.mark.asyncio
async def test_classify_errors_over_max_rows() -> None:
    t = A.Table(columns=["text"], rows=[{"text": str(i)} for i in range(5)])
    with pytest.raises(ValueError, match="exceeds max_rows"):
        await A.classify(t, column="text", labels=["a"], max_rows=3, llm=_StubLLM(labels="[]"))


@pytest.mark.asyncio
async def test_llm_op_needs_a_backend() -> None:
    # Through the executor with no llm injected → a clear "needs an LLM" error (not a crash).
    from vibe.core.graph.executor import GraphValidationError, execute

    g = Graph()
    g.add(Node(id="s", op="sample_dataset", params={"name": "sales"}))
    g.add(Node(id="n", op="narrate", params={"goal": "x"}, inputs={"table": "s"}))
    with pytest.raises(GraphValidationError, match="needs an LLM"):
        await execute(g, None, llm=None)


def test_llm_ops_hide_the_reserved_param() -> None:
    # `llm` is executor-injected, so it must not appear as a graph param (catalog/validate/fp).
    from vibe.core.graph.operators import get_operator

    for name in ("narrate", "classify"):
        spec = get_operator(name)
        assert spec.needs_llm and "llm" not in spec.param_names and "llm" not in spec.arg_types


@pytest.mark.asyncio
async def test_describe_output_is_sql_selectable() -> None:
    # Regression: the describe/correlation label column must not be a SQL reserved word — a
    # downstream sql() selecting it used to fail on `column`. `field` selects cleanly.
    t = await _sales()
    desc = await A.describe(t, columns=["revenue", "cost"])
    out = await A.sql(query="SELECT field, mean FROM t1 ORDER BY mean DESC", t1=desc)
    assert out.columns == ["field", "mean"]
    assert {r["field"] for r in out.rows} == {"revenue", "cost"}
    corr = await A.correlation(t, columns=["revenue", "cost"])
    assert "field" in corr.columns and "column" not in corr.columns


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "call"),
    [
        ("scatter", lambda t, p: A.chart(t, kind="scatter", x="units", y="revenue", path=p)),
        ("histogram", lambda t, p: A.chart(t, kind="histogram", column="revenue", path=p)),
        ("box", lambda t, p: A.chart(t, kind="box", column="revenue", by="region", path=p)),
        ("pie", lambda t, p: A.chart(t, kind="pie", labels="region", values="revenue", path=p)),
        ("line", lambda t, p: A.chart(t, kind="line", x="date", y="revenue", path=p, series="region")),
    ],
)
async def test_chart_ops_write_png_and_return_handle(kind, call, tmp_path) -> None:
    out = tmp_path / f"{kind}.png"
    res = await call(await _sales(), str(out))
    assert isinstance(res, A.ChartResult) and res.kind == kind
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.asyncio
async def test_heatmap_of_correlation(tmp_path) -> None:
    corr = await A.correlation(await _sales(), columns=["units", "revenue", "cost"])
    out = tmp_path / "hm.png"
    res = await A.chart(corr, kind="heatmap", path=str(out))
    assert res.kind == "heatmap" and out.exists()


@pytest.mark.asyncio
async def test_chart_bad_axis_errors(tmp_path) -> None:
    # a non-numeric axis on a numeric-only chart fails with a clear, column-naming error
    with pytest.raises(ValueError, match="not numeric"):
        await A.chart(await _sales(), kind="scatter", x="region", y="revenue", path=str(tmp_path / "x.png"))


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_feature_importance_ranks_and_aggregates() -> None:
    t = await _sales()
    fi = await A.feature_importance(
        t, target="region", features=["revenue", "units", "cost", "product"], model="rf"
    )
    assert fi.columns == ["field", "importance"]
    # one-hot columns (product_*) are summed back to the source feature the caller named
    assert {r["field"] for r in fi.rows} == {"revenue", "units", "cost", "product"}
    imps = [r["importance"] for r in fi.rows]
    assert imps == sorted(imps, reverse=True)  # ranked
    # a numeric target auto-infers regression; linear on a class target is rejected clearly
    reg = await A.feature_importance(t, target="revenue", features=["units", "cost"], model="linear")
    assert {r["field"] for r in reg.rows} == {"units", "cost"}
    with pytest.raises(ValueError, match="regression target"):
        await A.feature_importance(t, target="region", features=["revenue"], model="linear")


# --- inferential statistics: scipy reference-parity ----------------------------------------


@pytest.mark.asyncio
async def test_corr_test_matches_scipy() -> None:
    from scipy import stats

    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    ys = [2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.7]
    t = A.Table(columns=["x", "y"], rows=[{"x": x, "y": y} for x, y in zip(xs, ys, strict=True)])
    out = await A.corr_test(t, x="x", y="y", method="pearson")
    r_exp, p_exp = stats.pearsonr(xs, ys)
    assert out.rows[0]["coefficient"] == pytest.approx(float(r_exp), rel=1e-9)
    # p_value: abs tolerance — the JSON wire format caps ~10 decimal places, so tiny p-values lose
    # relative precision (harmless: the benchmark rounds p to 4 decimals and significance is p<α).
    assert out.rows[0]["p_value"] == pytest.approx(float(p_exp), abs=1e-9)
    assert out.rows[0]["n"] == len(xs)
    # spearman path also matches
    sp = await A.corr_test(t, x="x", y="y", method="spearman")
    assert sp.rows[0]["coefficient"] == pytest.approx(float(stats.spearmanr(xs, ys)[0]), rel=1e-9)
    with pytest.raises(ValueError, match="≥3 complete"):
        await A.corr_test(A.Table(columns=["x", "y"], rows=[{"x": 1, "y": 2}]), x="x", y="y")


@pytest.mark.asyncio
async def test_group_test_matches_scipy() -> None:
    from scipy import stats

    a = [10.0, 12.0, 11.0, 13.0, 9.0]
    b = [20.0, 22.0, 19.0, 21.0, 23.0]
    rows = [{"v": v, "g": "a"} for v in a] + [{"v": v, "g": "b"} for v in b]
    t = A.Table(columns=["v", "g"], rows=rows)
    tt = await A.group_test(t, value="v", group="g", test="ttest")
    exp = stats.ttest_ind(a, b, equal_var=True)
    assert tt.rows[0]["statistic"] == pytest.approx(float(exp[0]), rel=1e-9)
    assert tt.rows[0]["p_value"] == pytest.approx(float(exp[1]), abs=1e-9)  # abs: see corr_test note
    assert tt.rows[0]["n_groups"] == 2
    welch = await A.group_test(t, value="v", group="g", test="welch")
    assert welch.rows[0]["p_value"] == pytest.approx(float(stats.ttest_ind(a, b, equal_var=False)[1]), abs=1e-9)
    mw = await A.group_test(t, value="v", group="g", test="mannwhitney")
    assert mw.rows[0]["p_value"] == pytest.approx(float(stats.mannwhitneyu(a, b)[1]), abs=1e-9)
    # a three-group frame: anova works, ttest errors on arity
    t3 = A.Table(columns=["v", "g"], rows=rows + [{"v": v, "g": "c"} for v in [30.0, 31.0, 29.0]])
    an = await A.group_test(t3, value="v", group="g", test="anova")
    assert an.rows[0]["n_groups"] == 3
    with pytest.raises(ValueError, match="exactly 2 groups"):
        await A.group_test(t3, value="v", group="g", test="ttest")


@pytest.mark.asyncio
async def test_normality_and_chi_square_match_scipy() -> None:
    from scipy import stats

    vals = [2.1, 3.4, 1.9, 5.6, 4.2, 3.3, 2.8, 4.9, 3.1, 2.2, 5.0, 3.7]
    t = A.Table(columns=["v"], rows=[{"v": v} for v in vals])
    sh = await A.normality_test(t, column="v", method="shapiro")
    assert sh.rows[0]["p_value"] == pytest.approx(float(stats.shapiro(vals)[1]), rel=1e-9)
    assert sh.rows[0]["critical_value"] is None
    an = await A.normality_test(t, column="v", method="anderson")
    assert an.rows[0]["critical_value"] is not None and an.rows[0]["p_value"] is None
    with pytest.raises(ValueError, match="shapiro needs"):
        await A.normality_test(A.Table(columns=["v"], rows=[{"v": 1.0}, {"v": 2.0}]), column="v")

    ct = A.Table(
        columns=["sex", "survived"],
        rows=[{"sex": s, "survived": v} for s, v in
              [("m", "no")] * 8 + [("m", "yes")] * 2 + [("f", "no")] * 3 + [("f", "yes")] * 7],
    )
    import pandas as pd

    chi = await A.chi_square(ct, column1="sex", column2="survived")
    exp = stats.chi2_contingency(pd.crosstab([r["sex"] for r in ct.rows], [r["survived"] for r in ct.rows]))
    assert chi.rows[0]["statistic"] == pytest.approx(float(exp[0]), rel=1e-9)
    assert chi.rows[0]["dof"] == int(exp[2])


def _confounded_regression_table() -> A.Table:
    # A textbook Simpson's-paradox setup: within each pclass, fare clearly *decreases* with age
    # (slope -5), but pclass 1 (old passengers) is also the most expensive class overall — so the
    # unconditional/bivariate age-fare trend is strongly *positive*, flipping sign once pclass is
    # controlled for. Verified numerically: bivariate corr(age, fare) ≈ +0.99, but the true
    # partial coefficient of age (holding pclass fixed) is exactly -5.0.
    rows = []
    for pclass, base_fare, ages in (
        (1, 200.0, [68, 69, 70, 71, 72]),
        (2, 100.0, [38, 39, 40, 41, 42]),
        (3, 20.0, [8, 9, 10, 11, 12]),
    ):
        mean_age = sum(ages) / len(ages)
        for age in ages:
            rows.append({"age": float(age), "pclass": float(pclass), "fare": base_fare - 5.0 * (age - mean_age)})
    return A.Table(columns=["age", "pclass", "fare"], rows=rows)


@pytest.mark.asyncio
async def test_regression_summary_matches_statsmodels_ols() -> None:
    import statsmodels.api as sm

    t = _confounded_regression_table()
    df = A._to_df(t)
    x = sm.add_constant(df[["age", "pclass"]])
    ref = sm.OLS(df["fare"], x).fit()

    got = await A.regression_summary(t, target="fare", features=["age", "pclass"])
    by_field = {r["field"]: r for r in got.rows}

    assert by_field["Intercept"]["coef"] == pytest.approx(ref.params["const"], rel=1e-6)
    assert by_field["age"]["coef"] == pytest.approx(ref.params["age"], rel=1e-6)
    assert by_field["pclass"]["coef"] == pytest.approx(ref.params["pclass"], rel=1e-6)
    assert by_field["age"]["std_err"] == pytest.approx(ref.bse["age"], rel=1e-6)
    assert by_field["age"]["t_stat"] == pytest.approx(ref.tvalues["age"], rel=1e-6)
    assert by_field["age"]["p_value"] == pytest.approx(ref.pvalues["age"], rel=1e-6)


@pytest.mark.asyncio
async def test_regression_summary_controls_for_confounder_unlike_bivariate_corr() -> None:
    # The actual DABench failure mode: a bivariate corr_test(age, fare) sign-flips relative to
    # regression_summary's age coefficient once pclass is controlled for.
    t = _confounded_regression_table()
    bivariate = await A.corr_test(t, x="age", y="fare")
    controlled = await A.regression_summary(t, target="fare", features=["age", "pclass"])
    age_coef = next(r["coef"] for r in controlled.rows if r["field"] == "age")
    assert (bivariate.rows[0]["coefficient"] > 0) != (age_coef > 0)


@pytest.mark.asyncio
async def test_regression_summary_intercept_false_omits_intercept_row() -> None:
    t = _confounded_regression_table()
    got = await A.regression_summary(t, target="fare", features=["age", "pclass"], intercept=False)
    assert [r["field"] for r in got.rows] == ["age", "pclass"]


@pytest.mark.asyncio
async def test_regression_summary_rejects_non_numeric_feature() -> None:
    t = A.Table(columns=["x", "y"], rows=[{"x": "a", "y": 1.0}, {"x": "b", "y": 2.0}, {"x": "c", "y": 3.0}])
    with pytest.raises(ValueError, match="not numeric"):
        await A.regression_summary(t, target="y", features=["x"])


@pytest.mark.asyncio
async def test_regression_summary_raises_on_collinear_features() -> None:
    # x2 is an exact linear function of x — the design matrix is singular even with plenty of
    # rows to spare for degrees of freedom, so this must fail on collinearity, not row count.
    t = A.Table(
        columns=["x", "x2", "y"],
        rows=[{"x": float(i), "x2": float(2 * i), "y": float(i)} for i in range(1, 8)],
    )
    with pytest.raises(ValueError, match="collinear"):
        await A.regression_summary(t, target="y", features=["x", "x2"])


@pytest.mark.asyncio
async def test_regression_summary_raises_on_insufficient_dof() -> None:
    t = A.Table(columns=["x", "y"], rows=[{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 2.0}])
    with pytest.raises(ValueError, match="degrees of freedom"):
        await A.regression_summary(t, target="y", features=["x"])


# --- preprocessing transforms --------------------------------------------------------------


@pytest.mark.asyncio
async def test_normalize_minmax_and_zscore() -> None:
    t = A.Table(columns=["v"], rows=[{"v": 0.0}, {"v": 5.0}, {"v": 10.0}])
    mm = await A.normalize(t, column="v", method="minmax")
    assert [r["v"] for r in mm.rows] == pytest.approx([0.0, 0.5, 1.0])
    zs = await A.normalize(t, column="v", method="zscore", name="z")
    zcol = [r["z"] for r in zs.rows]
    assert sum(zcol) == pytest.approx(0.0, abs=1e-9) and "z" in zs.columns
    # constant column → all zeros (no divide-by-zero)
    const = await A.normalize(A.Table(columns=["v"], rows=[{"v": 3.0}, {"v": 3.0}]), column="v")
    assert [r["v"] for r in const.rows] == [0.0, 0.0]


@pytest.mark.asyncio
async def test_encode_label_and_onehot() -> None:
    t = A.Table(columns=["g"], rows=[{"g": "b"}, {"g": "a"}, {"g": "a"}, {"g": "c"}])
    lab = await A.encode(t, column="g", method="label")
    assert [r["g"] for r in lab.rows] == [1, 0, 0, 2]  # sorted: a=0, b=1, c=2
    oh = await A.encode(t, column="g", method="onehot")
    assert {"g_a", "g_b", "g_c"} <= set(oh.columns) and "g" not in oh.columns
    assert all(r["g_a"] in (0, 1) for r in oh.rows)  # ints, not booleans


@pytest.mark.asyncio
async def test_outliers_zscore_and_fill_mode() -> None:
    t = A.Table(columns=["v"], rows=[{"v": float(i)} for i in range(20)] + [{"v": 1000.0}])
    oz = await A.outliers(t, column="v", method="zscore", factor=3.0)
    assert oz.rows[0]["count"] == 1 and oz.rows[0]["total"] == 21
    fm = await A.fill_missing(
        A.Table(columns=["c"], rows=[{"c": "a"}, {"c": "a"}, {"c": None}]), column="c", method="mode"
    )
    assert [r["c"] for r in fm.rows] == ["a", "a", "a"]


@pytest.mark.asyncio
async def test_ml_encode_label_runs_and_differs_from_onehot() -> None:
    t = await _sales()
    lab = await A.ml_regression(
        t, target="revenue", features=["units", "region"], model="linear", metric="mse",
        evaluate="full", encode="label",
    )
    oh = await A.ml_regression(
        t, target="revenue", features=["units", "region"], model="linear", metric="mse",
        evaluate="full", encode="onehot",
    )
    assert lab.columns == ["score"] and oh.columns == ["score"]
    # label-encoding a multi-value categorical yields a different fit than one-hot
    assert lab.rows[0]["score"] != oh.rows[0]["score"]


def test_catalog_is_grouped_by_category() -> None:

    from vibe.core.graph.operators import registered_operators
    from vibe.core.graph.render import _CATEGORY_ORDER, operators_catalog

    ops = {n: s for n, s in registered_operators().items() if s.library == "analysis"}
    # every analysis op is tagged with a known category (no stragglers in "other")
    assert ops and all(s.category in _CATEGORY_ORDER for s in ops.values())
    cat = operators_catalog(ops, {}, verbose=True)
    # category headers render in the defined order
    seen = [c for c in _CATEGORY_ORDER if f"[{c}]" in cat]
    assert seen == [c for c in _CATEGORY_ORDER if c in {s.category for s in ops.values()}]
    positions = [cat.index(f"[{c}]") for c in seen]
    assert positions == sorted(positions)
    assert "[statistics]" in cat and "[ml]" in cat and "[inference]" in cat


def test_untagged_catalog_stays_flat() -> None:
    # Backward-compatible: a catalog whose ops carry no category renders as a flat sorted list (no
    # headers) — so untagged libraries are unchanged. Clone analysis specs with category stripped.
    import dataclasses
    import re

    from vibe.core.graph.operators import registered_operators
    from vibe.core.graph.render import operators_catalog

    untagged = {
        n: dataclasses.replace(s, category=None)
        for n, s in registered_operators().items()
        if s.library == "analysis"
    }
    op_section = operators_catalog(untagged, {}, verbose=False).split("Available blocks")[0]
    # no bare "[category]" header lines when nothing is tagged
    assert not re.search(r"(?m)^\[\w+\]$", op_section)
