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

    out = await A.outliers_iqr(t, column="revenue")
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
    chart = await A.bar_chart(ranked, x="country", y="revenue_sum", path=str(png), title="Top")
    assert png.exists() and png.stat().st_size > 0 and chart.kind == "bar"
    # the value stays a tiny handle, not the image bytes
    assert set(chart.model_dump()) == {"path", "kind"}
    line_png = tmp_path / "line.png"
    line = await A.line_chart(ranked, x="country", y="revenue_sum", path=str(line_png), title="Trend")
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


@pytest.mark.timeout(60)  # cold sklearn/scipy import + fits, tolerant of CI/xdist contention
@pytest.mark.asyncio
async def test_ml_operators_run_and_are_seeded_deterministic() -> None:
    # Skipped unless the optional [ml] extra is installed (uv run --extra ml). Cheap models keep it
    # fast; determinism is what matters for cache correctness.
    pytest.importorskip("sklearn")
    t = await _sales()

    reg = await A.ml_regression(t, target="revenue", features=["units", "cost"], model="linear", metric="r2")
    assert reg.columns == ["score"] and isinstance(reg.rows[0]["score"], (int, float))

    clf1 = await A.ml_classification(t, target="region", features=["revenue", "units", "cost"], model="tree")
    clf2 = await A.ml_classification(t, target="region", features=["revenue", "units", "cost"], model="tree")
    assert clf1.rows[0]["score"] == clf2.rows[0]["score"]  # fixed seed → deterministic

    clus = await A.ml_cluster(t, features=["revenue", "units"], k=2)
    assert clus.columns == ["score"] and -1.0 <= clus.rows[0]["score"] <= 1.0  # silhouette range

    with pytest.raises(ValueError, match="model must be one of"):
        await A.ml_regression(t, target="revenue", features=["units"], model="nope")


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
