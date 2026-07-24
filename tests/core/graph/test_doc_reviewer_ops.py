from __future__ import annotations

from pathlib import Path

import pytest

from vibe.core.graph.executor import GraphValidationError, execute
from vibe.core.graph.fingerprint import content_hash
import vibe.core.graph.library.doc_reviewer as D
from vibe.core.graph.model import Graph, Node


def _doc() -> D.Document:
    return D.Document(
        name="c",
        sections=[
            D.Section(heading="Limitation of Liability", text="Aggregate liability is capped at fees."),
            D.Section(heading="Termination", text="Either party may terminate for material breach."),
        ],
    )


class _StubLLM:
    """A deterministic stand-in for the injected LLMCaller — returns a canned reply, no backend."""

    def __init__(self, reply: str) -> None:
        self.reply, self.calls = reply, 0

    async def __call__(self, prompt: str, *, system: str | None = None, max_tokens: int = 1024) -> str:
        self.calls += 1
        return self.reply


class _SeqStubLLM:
    """Returns queued replies in order (to exercise the empty-then-retry path of _llm_json)."""

    def __init__(self, *replies: str) -> None:
        self._replies = list(replies)
        self.calls = 0

    async def __call__(self, prompt: str, *, system: str | None = None, max_tokens: int = 1024) -> str:
        out = self._replies[min(self.calls, len(self._replies) - 1)]
        self.calls += 1
        return out


# --- pure ops ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_document_segments_on_headings(tmp_path: Path) -> None:
    p = tmp_path / "c.md"
    p.write_text("Preamble text.\n# Section One\nBody one.\n## 2. Section Two\nBody two.\n")
    doc = await D.read_document(path=str(p), content_fp=content_hash(str(p)))
    headings = [s.heading for s in doc.sections]
    assert "Section One" in headings and "2. Section Two" in headings
    assert doc.sections[0].heading == "Preamble"


@pytest.mark.asyncio
async def test_read_document_pdf_uses_pypdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "c.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(D, "_read_pdf", lambda path: "# Termination\nEither party may terminate.")
    doc = await D.read_document(path=str(p), content_fp=content_hash(str(p)))
    assert [s.heading for s in doc.sections] == ["Termination"]


@pytest.mark.asyncio
async def test_read_document_pdf_without_pypdf_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "c.pdf"
    p.write_bytes(b"%PDF-1.4 fake")

    def _boom() -> None:
        raise ValueError("read_document: reading a PDF needs pypdf")

    monkeypatch.setattr(D, "_require_pypdf", _boom)
    with pytest.raises(ValueError, match="needs pypdf"):
        await D.read_document(path=str(p), content_fp=content_hash(str(p)))


@pytest.mark.asyncio
async def test_sample_document_loads_and_unknown_errors() -> None:
    doc = await D.sample_document(name="msa")
    assert any("Termination" in s.heading for s in doc.sections)
    assert len(doc.sections) >= 5
    with pytest.raises(ValueError, match="unknown sample document"):
        await D.sample_document(name="nope")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_clean_document_strips_tags_and_drops_empty() -> None:
    doc = D.Document(sections=[
        D.Section(heading="<PAGE>", text=""),
        D.Section(heading="1. Term", text="<b>Twelve</b> months <br/>only."),
    ])
    out = await D.clean_document(doc)
    assert [s.heading for s in out.sections] == ["1. Term"]
    assert out.sections[0].text == "Twelve months only."


@pytest.mark.asyncio
async def test_chunk_document_merges_and_caps() -> None:
    doc = D.Document(sections=[D.Section(heading=f"h{i}", text="word " * 10) for i in range(10)])
    out = await D.chunk_document(doc, max_sections=3)
    assert len(out.sections) <= 3
    # a tiny fragment merges into the previous section
    doc2 = D.Document(sections=[
        D.Section(heading="Real", text="this is a full sentence with many words here"),
        D.Section(heading="", text="12"),
    ])
    out2 = await D.chunk_document(doc2, max_sections=80)
    assert len(out2.sections) == 1 and "12" in out2.sections[0].text


@pytest.mark.asyncio
async def test_search_and_select_sections() -> None:
    doc = _doc()
    assert [s.heading for s in (await D.search_sections(doc, r"liab\w+")).sections] == ["Limitation of Liability"]
    assert [s.heading for s in (await D.select_sections(doc, ["termination"])).sections] == ["Termination"]
    with pytest.raises(ValueError, match="invalid regex"):
        await D.search_sections(doc, "[")


@pytest.mark.asyncio
async def test_expect_sections_guards() -> None:
    assert (await D.expect_sections(_doc(), min_count=2)).sections
    with pytest.raises(ValueError, match="fewer than the expected"):
        await D.expect_sections(_doc(), min_count=5)


@pytest.mark.asyncio
async def test_find_missing_is_pure() -> None:
    items = D.ItemSet(items=[D.Item(category="termination")])
    rep = await D.find_missing(items, required=["termination", "arbitration"])
    assert "- arbitration" in rep.markdown


@pytest.mark.asyncio
async def test_check_references_flags_unresolved() -> None:
    doc = D.Document(sections=[
        D.Section(heading="1. Term", text="See Section 1 and Section 9.9 for details."),
    ])
    rep = await D.check_references(doc)
    assert "§9.9" in rep.markdown and "§1" not in rep.markdown  # 1 exists, 9.9 does not


@pytest.mark.asyncio
async def test_check_definitions_flags_terms() -> None:
    doc = D.Document(sections=[
        D.Section(heading="Defs", text='"Widget" means a thing. The "Gadget" is used but undefined.'),
    ])
    rep = await D.check_definitions(doc)
    assert "Widget" in rep.markdown  # defined but not used again
    assert "Gadget" in rep.markdown  # used but not defined


@pytest.mark.asyncio
async def test_outline_and_items_to_markdown() -> None:
    outline = await D.outline(_doc(), title="Outline")
    assert outline.markdown.startswith("# Outline") and "Limitation of Liability" in outline.markdown
    md = await D.items_to_markdown(
        D.ItemSet(items=[D.Item(section="Termination", category="termination", labels={"risk": "high"})])
    )
    assert "| Section |" in md.markdown and "risk=high" in md.markdown


@pytest.mark.asyncio
async def test_combine_reports_merges() -> None:
    out = await D.combine_reports(D.Report(markdown="# A\naaa"), D.Report(markdown="# B\nbbb"), title="Packet")
    assert "# Packet" in out.markdown and "aaa" in out.markdown and "bbb" in out.markdown


# --- LLM-backed ops ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_segments_builds_items() -> None:
    llm = _StubLLM('[{"section": 0, "category": "liability"}, {"section": 1, "category": "termination"}]')
    out = await D.extract_segments(_doc(), categories=["liability", "termination"], llm=llm)
    assert [(i.section, i.category) for i in out.items] == [
        ("Limitation of Liability", "liability"),
        ("Termination", "termination"),
    ]


@pytest.mark.asyncio
async def test_extract_segments_rejects_bad_output() -> None:
    with pytest.raises(ValueError, match="not in"):
        await D.extract_segments(_doc(), categories=["liability"], llm=_StubLLM('[{"section": 0, "category": "zzz"}]'))
    with pytest.raises(ValueError, match="index out of range"):
        await D.extract_segments(_doc(), categories=["liability"], llm=_StubLLM('[{"section": 9, "category": "liability"}]'))


@pytest.mark.asyncio
async def test_extract_fields_maps_field_to_category() -> None:
    llm = _StubLLM('[{"field": "term", "value": "12 months", "section": 1}]')
    out = await D.extract_fields(_doc(), fields=["term", "parties"], llm=llm)
    assert [(i.category, i.text) for i in out.items] == [("term", "12 months")]


@pytest.mark.asyncio
async def test_classify_writes_dimension_label() -> None:
    items = D.ItemSet(items=[D.Item(category="liability", text="uncapped"), D.Item(category="term", text="mutual")])
    out = await D.classify(
        items, dimension="risk", labels=["low", "medium", "high"],
        llm=_StubLLM('[{"label": "high", "note": "uncapped"}, {"label": "low", "note": "ok"}]'),
    )
    assert [i.labels["risk"] for i in out.items] == ["high", "low"]
    # a second classify stacks another dimension
    out2 = await D.classify(
        out, dimension="favorability", labels=["favorable", "neutral", "unfavorable"],
        llm=_StubLLM('[{"label": "unfavorable", "note": "x"}, {"label": "neutral", "note": "y"}]'),
    )
    assert out2.items[0].labels == {"risk": "high", "favorability": "unfavorable"}


@pytest.mark.asyncio
async def test_classify_rejects_bad_label() -> None:
    items = D.ItemSet(items=[D.Item(category="x")])
    with pytest.raises(ValueError, match="label"):
        await D.classify(items, dimension="risk", labels=["low", "high"], llm=_StubLLM('[{"label": "extreme"}]'))


@pytest.mark.asyncio
async def test_compare_to_reference_reads_file(tmp_path: Path) -> None:
    ref = tmp_path / "ref.txt"
    ref.write_text("Payment: Net 30 is standard.")
    items = D.ItemSet(items=[D.Item(category="payment", text="Net 45"), D.Item(category="term", text="12 months")])
    llm = _StubLLM('[{"flag": true, "note": "Net 45 > Net 30"}, {"flag": false, "note": "ok"}]')
    out = await D.compare_to_reference(items, reference_path=str(ref), content_fp="", llm=llm)
    assert out.items[0].note.startswith("flagged:") and out.items[1].note.startswith("ok:")


@pytest.mark.asyncio
async def test_answer_summarize_suggest_write_return_prose() -> None:
    assert (await D.answer_question(_doc(), question="term?", llm=_StubLLM("2 years."))).markdown == "2 years."
    assert (await D.summarize_document(_doc(), llm=_StubLLM("A summary."))).markdown == "A summary."
    items = D.ItemSet(items=[D.Item(category="liability", labels={"risk": "high"})])
    assert (await D.suggest_edits(items, llm=_StubLLM("- Cap it."))).markdown == "- Cap it."
    assert (await D.write_memo(items, llm=_StubLLM("## Executive summary\nx"))).markdown.startswith("## Executive")


@pytest.mark.asyncio
async def test_llm_json_retries_then_errors_on_empty() -> None:
    doc = _doc()
    # empty first, valid on the retry → succeeds
    seq = _SeqStubLLM("", '[{"section": 0, "category": "liability"}]')
    out = await D.extract_segments(doc, categories=["liability"], llm=seq)
    assert len(out.items) == 1 and seq.calls == 2
    # always empty → actionable error (the compare_to_playbook failure the user hit)
    with pytest.raises(ValueError, match="returned no content"):
        await D.extract_segments(doc, categories=["liability"], llm=_StubLLM(""))


# --- executor / registry contract ----------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_op_needs_a_backend() -> None:
    g = Graph()
    g.add(Node(id="d", op="sample_document", params={"name": "msa"}))
    g.add(Node(id="e", op="extract_segments", params={"categories": ["termination"]}, inputs={"doc": "d"}))
    with pytest.raises(GraphValidationError, match="needs an LLM"):
        await execute(g, None, llm=None)


def test_llm_ops_hide_the_reserved_param() -> None:
    from vibe.core.graph.operators import get_operator

    for name in ("extract_segments", "extract_fields", "classify", "compare_to_reference",
                 "answer_question", "summarize_document", "suggest_edits", "write_memo"):
        spec = get_operator(name)
        assert spec.needs_llm and "llm" not in spec.param_names and "llm" not in spec.arg_types


def test_blocks_are_registered() -> None:
    from vibe.core.graph.blocks import is_block

    assert all(is_block(n) for n in ("inventory", "contract_review", "term_sheet", "paper_abstract", "gap_check"))
