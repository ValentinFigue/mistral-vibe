from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from vibe.core.graph.operators import coerce_value, get_operator, operator, param_issues


class _Row(BaseModel):
    v: int


class _Agg(BaseModel):
    n: int


@operator(name="tst_meta")
async def _meta(rows: list[_Row], label: str) -> _Agg:
    """Aggregate rows into a count."""
    return _Agg(n=len(rows))


def test_operator_metadata_arg_types_and_description() -> None:
    spec = get_operator("tst_meta")
    # list[Model] is recognized as a wired input; the scalar is a param.
    assert "rows" in spec.input_names
    assert "label" not in spec.input_names
    assert spec.literal_names() == ("label",)
    # arg types are captured, readable.
    assert spec.arg_types["rows"].startswith("list[")
    assert spec.arg_types["label"] == "str"
    # description is the first docstring line.
    assert spec.description == "Aggregate rows into a count."


@operator(name="tst_blank_doc")
async def _blank_doc(x: _Row) -> _Agg:
    """   """  # whitespace-only docstring must not crash registration
    return _Agg(n=x.v)


def test_whitespace_only_docstring_registers() -> None:
    spec = get_operator("tst_blank_doc")
    assert spec.description == ""


def test_demo_operator_metadata() -> None:
    from vibe.core.graph.demo import pipeline  # noqa: F401 — registers demo ops

    spec = get_operator("parse_table")
    assert spec.input_names == ("file",)
    assert spec.arg_types == {"file": "FileContent", "amount_col": "str"}
    assert spec.description


@operator(name="tst_lib", library="mylib", reads_file="path")
async def _lib_op(path: str, content_fp: str) -> _Agg:
    """A tagged, file-reading operator."""
    return _Agg(n=1)


def test_operator_library_and_reads_file_tags() -> None:
    spec = get_operator("tst_lib")
    assert spec.library == "mylib"
    assert spec.reads_file == "path"
    # untagged ops default to None (shown to the generic agent)
    assert get_operator("tst_meta").library is None


def test_reads_file_must_be_a_param() -> None:
    import pytest

    with pytest.raises(TypeError, match="not a parameter"):

        @operator(name="tst_bad_reads", reads_file="missing")
        async def _bad(x: _Row) -> _Agg:
            return _Agg(n=x.v)


@operator(name="tst_enum")
async def _enum(
    x: _Row,
    mode: Literal["fast", "slow"] = "fast",
    tags: list[Literal["a", "b"]] | None = None,
    n: int = 1,
) -> _Agg:
    """Op with an enum, a list-enum, and an int param."""
    return _Agg(n=x.v)


def test_literal_enum_metadata_and_rendering() -> None:
    spec = get_operator("tst_enum")
    # Literal renders as the allowed values in the catalog string, and is extracted structurally.
    assert spec.arg_types["mode"] == "fast|slow"
    assert spec.allowed_values["mode"] == ("fast", "slow")
    assert spec.allowed_values["tags"] == ("a", "b")  # Optional[list[Literal]] unwrapped
    assert spec.defaults["mode"] == "fast" and spec.defaults["mode"] in spec.allowed_values["mode"]
    assert "mode" in spec.param_types  # structured type kept for coercion/checks


def test_param_issues_enum_type_and_hint() -> None:
    spec = get_operator("tst_enum")
    bad = param_issues(spec, {"mode": "medium"})
    assert bad and "must be one of fast|slow" in bad[0] and "medium" in bad[0]
    assert "did you mean 'slow'" in param_issues(spec, {"mode": "sl0w"})[0]  # close typo → hint
    assert param_issues(spec, {"tags": ["a", "z"]})  # list-enum element checked
    assert any("expected int" in i for i in param_issues(spec, {"n": "abc"}))
    assert param_issues(spec, {"mode": "fast", "tags": ["a"], "n": 3}) == []  # all valid


def test_coerce_value_safe_cases() -> None:
    assert coerce_value("10", int) == 10
    assert coerce_value("0.9", float) == 0.9
    assert coerce_value("true", bool) is True
    assert coerce_value(5, str) == "5"
    assert coerce_value("country", list[str]) == ["country"]  # scalar → one-element list
    assert coerce_value(["a", "b"], list[str]) == ["a", "b"]
    assert coerce_value("abc", int) == "abc"  # non-numeric: left for validate to reject
