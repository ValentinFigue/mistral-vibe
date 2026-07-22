from __future__ import annotations

import json

import pytest

from vibe.core.graph.cache import CacheStore
from vibe.core.graph.dsl import DSLError, parse_pipeline
from vibe.core.graph.executor import execute
import vibe.core.graph.library.analysis  # noqa: F401 — registers the analysis operators/blocks


def test_linear_chain_auto_wires() -> None:
    g = parse_pipeline(
        'sample_dataset(name="sales")'
        ' | filter_rows(column="product", value="widget")'
        ' | group_by(keys=["country"], metric="revenue", aggs=["sum"])'
        ' | top_n(by="revenue_sum", n=3)'
        ' | to_markdown(title="Top")'
    )
    assert list(g.nodes) == ["s1", "s2", "s3", "s4", "s5"]
    assert g.nodes["s1"].op == "sample_dataset" and not g.nodes["s1"].inputs  # source, no piped input
    assert g.nodes["s2"].inputs == {"table": "s1"}  # piped into first input port
    assert g.nodes["s4"].params == {"by": "revenue_sum", "n": 3}


def test_multiline_pipeline_folds_continuation_lines() -> None:
    # The readable multi-line form (a leading `|` continues the previous line) parses to the
    # same graph as the single-line form — this is what the analyst prompt teaches agents to write.
    multi = parse_pipeline(
        '\n'
        'read_csv(path="orders.csv")\n'
        '  | expect_no_nulls(columns=["revenue", "cost"])\n'
        '  | sql(query="SELECT channel, sum(revenue) r FROM t1 GROUP BY channel ORDER BY r DESC")\n'
        '  | to_markdown(title="By channel")\n'
    )
    single = parse_pipeline(
        'read_csv(path="orders.csv") '
        '| expect_no_nulls(columns=["revenue", "cost"]) '
        '| sql(query="SELECT channel, sum(revenue) r FROM t1 GROUP BY channel ORDER BY r DESC") '
        '| to_markdown(title="By channel")'
    )
    assert list(multi.nodes) == list(single.nodes) == ["s1", "s2", "s3", "s4"]
    assert multi.nodes["s2"].inputs == {"table": "s1"}
    assert multi.nodes["s3"].inputs == {"t1": "s2"}


def test_multiline_triple_quoted_sql_body() -> None:
    # The shape agents actually write: a multi-line pipeline whose sql(query="""...""") body
    # spans several lines. Newlines inside the triple-quoted string must NOT split the statement.
    g = parse_pipeline(
        'read_csv(path="orders.csv")\n'
        '  | expect_no_nulls(columns=["revenue", "cost"])\n'
        '  | sql(query="""\n'
        "      SELECT channel, sum(revenue) AS r\n"
        "      FROM t1\n"
        "      WHERE channel = 'web'\n"
        "      GROUP BY 1\n"
        '  """)\n'
        '  | to_markdown(title="X")'
    )
    assert list(g.nodes) == ["s1", "s2", "s3", "s4"]
    q = g.nodes["s3"].params["query"]
    assert "channel = 'web'" in q and "GROUP BY 1" in q and "\n" in q  # body kept, incl. newlines
    assert g.nodes["s3"].inputs == {"t1": "s2"}


# A batch of realistic agent-authored programs — the parser must accept every one. Grown from
# actual live failures; add a case here whenever a new agent-written form is found to break.
_REALISTIC_PROGRAMS = [
    # trailing-`|` continuation (pipe at end of line, not start of next)
    'read_csv(path="x.csv") |\n sql(query="SELECT * FROM t1") |\n to_markdown()',
    # leading `#` comment + blank lines + indented continuation
    '# revenue report\n\nsample_dataset(name="sales")\n  | to_markdown(title="X")\n',
    # SQL string concat (||) and LIKE % inside a triple-quoted query
    'sample_dataset(name="sales") | sql(query="""SELECT country||region AS cr'
    " FROM t1 WHERE country LIKE 'f%'\"\"\") | to_markdown()",
    # aggregate parens + commas inside SQL (must not split args or trip paren depth)
    'sample_dataset(name="sales") | sql(query="""SELECT count(*), sum(revenue)'
    ' FROM t1 GROUP BY 1, 2""") | to_markdown()',
    # SQL -- line comment inside a multi-line triple-quoted body
    'sample_dataset(name="sales") | sql(query="""SELECT a -- the col\nFROM t1""") | to_markdown()',
    # SQL # inside a triple-quoted body must stay literal (not treated as a DSL comment)
    'sample_dataset(name="sales") | sql(query="""SELECT a # kept\nFROM t1""") | to_markdown()',
    # triple-single-quoted multi-line body with an embedded single quote
    "sample_dataset(name=\"sales\") | sql(query='''\nSELECT a FROM t1\nWHERE a = 'x'\n''')"
    ' | to_markdown()',
    # single-quoted outer arg string
    "sample_dataset(name=\"sales\") | sql(query='SELECT * FROM t1 WHERE x = 5') | to_markdown()",
    # double-quoted identifiers inside a triple-quoted query
    'sample_dataset(name="sales") | sql(query="""SELECT "weird col" AS w FROM t1""")'
    ' | to_markdown()',
    # block used like an operator
    'sample_dataset(name="sales") | rank_by(group_key="country", metric="revenue", n=3, title="T")',
    # list + int params; a pure sink terminal
    'read_csv(path="x.csv") | expect_columns(columns=["a", "b"]) | to_csv(path="out.csv")',
]


@pytest.mark.parametrize("program", _REALISTIC_PROGRAMS)
def test_realistic_agent_programs_parse(program: str) -> None:
    g = parse_pipeline(program)
    assert len(g.nodes) >= 2  # every program builds a real multi-step graph


def test_triple_quoted_sql_carries_quotes_and_commas() -> None:
    # Agents embed SQL with """...""" to avoid escaping. The query holds single quotes, double
    # quotes, commas, and a pipe — none of which may trip the top-level splitter.
    g = parse_pipeline(
        'sample_dataset(name="sales")'
        ' | sql(query="""SELECT country, sum(revenue) AS r FROM t1'
        " WHERE country='fr' AND product <> 'a|b' GROUP BY 1, 2 ORDER BY r DESC\"\"\")"
        ' | to_markdown(title="X")'
    )
    assert list(g.nodes) == ["s1", "s2", "s3"]
    q = g.nodes["s2"].params["query"]
    assert "country='fr'" in q and "'a|b'" in q and "GROUP BY 1, 2" in q
    assert g.nodes["s2"].inputs == {"t1": "s1"} and g.nodes["s3"].inputs == {"table": "s2"}


def test_named_refs_and_sql_join_wiring() -> None:
    g = parse_pipeline(
        'orders = sample_dataset(name="sales")\n'
        'customers = sample_dataset(name="customers")\n'
        'orders | sql(query="SELECT * FROM t1 JOIN t2 USING(country)", t2=customers) | to_markdown()'
    )
    assert {"orders", "customers"} <= set(g.nodes)
    sql_node = next(n for n in g.nodes.values() if n.op == "sql")
    assert sql_node.inputs == {"t1": "orders", "t2": "customers"}  # pipe→t1, kwarg ref→t2
    assert sql_node.params["query"].startswith("SELECT")


@pytest.mark.asyncio
async def test_parsed_pipeline_executes(cache: CacheStore) -> None:
    g = parse_pipeline(
        'sample_dataset(name="sales") | rank_by(group_key="country", metric="revenue", n=3, title="R")'
    )
    values, _ = await execute(g, cache)
    term = next(nid for nid in values if nid.endswith("/r"))
    md = json.loads(cache.get(values[term].fingerprint).decode())["markdown"]
    assert md.startswith("# R") and "revenue_sum" in md  # block's computed template resolved


def test_literals_and_lists() -> None:
    g = parse_pipeline('sample_dataset(name="sales") | limit(n=5)')
    assert g.nodes["s2"].params == {"n": 5}
    g2 = parse_pipeline('sample_dataset(name="sales") | select_columns(columns=["a", "b"])')
    assert g2.nodes["s2"].params == {"columns": ["a", "b"]}


@pytest.mark.parametrize(
    ("program", "match"),
    [
        ('nope(x=1)', "unknown operator"),
        ('sample_dataset(name="sales") | read_csv(path="x")', "no piped input"),
        ('group_by(keys=missing_ref)', "not an input port"),
        ('orders | to_markdown()', "unknown reference 'orders'"),
        ('to_markdown(title=)', "cannot parse value"),
        ('to_markdown(title="x"', "expected `op"),
    ],
)
def test_parse_errors(program: str, match: str) -> None:
    with pytest.raises(DSLError, match=match):
        parse_pipeline(program)
