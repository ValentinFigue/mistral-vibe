from __future__ import annotations

from pydantic import BaseModel

from vibe.core.graph.operators import get_operator, operator


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
