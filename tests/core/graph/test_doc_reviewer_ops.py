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


# --- pure ops ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_document_segments_on_headings(tmp_path: Path) -> None:
    p = tmp_path / "c.md"
    p.write_text("Preamble text.\n# Section One\nBody one.\n## 2. Section Two\nBody two.\n")
    doc = await D.read_document(path=str(p), content_fp=content_hash(str(p)))
    headings = [s.heading for s in doc.sections]
    assert "Section One" in headings and "2. Section Two" in headings
    assert doc.sections[0].heading == "Preamble"  # text before the first heading


@pytest.mark.asyncio
async def test_read_document_pdf_uses_pypdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercise the .pdf branch without a real PDF: stub the pypdf reader the op imports.
    p = tmp_path / "c.pdf"
    p.write_bytes(b"%PDF-1.4 fake")

    class _Page:
        def extract_text(self) -> str:
            return "# Termination\nEither party may terminate."

    class _Reader:
        def __init__(self, _path: str) -> None:
            self.pages = [_Page()]

    monkeypatch.setattr(D, "_read_pdf", lambda path: _Reader(str(path)).pages[0].extract_text())
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


def test_blocks_are_registered() -> None:
    from vibe.core.graph.blocks import is_block

    assert all(is_block(name) for name in ("risk_review", "clause_inventory", "missing_clauses"))


@pytest.mark.asyncio
async def test_sample_contract_loads_and_unknown_errors() -> None:
    doc = await D.sample_contract(name="msa")
    assert any("Termination" in s.heading for s in doc.sections)
    assert len(doc.sections) >= 5
    with pytest.raises(ValueError, match="unknown sample contract"):
        await D.sample_contract(name="nope")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_filter_sections_matches_heading_or_text() -> None:
    out = await D.filter_sections(_doc(), contains="liability")
    assert [s.heading for s in out.sections] == ["Limitation of Liability"]


@pytest.mark.asyncio
async def test_expect_sections_guards_a_bad_parse() -> None:
    assert (await D.expect_sections(_doc(), min_count=2)).sections
    with pytest.raises(ValueError, match="fewer than the expected"):
        await D.expect_sections(_doc(), min_count=5)


@pytest.mark.asyncio
async def test_find_missing_clauses_is_pure() -> None:
    findings = D.Findings(items=[D.Finding(section="Termination", clause_type="termination")])
    rep = await D.find_missing_clauses(findings, required=["termination", "arbitration"])
    assert "- arbitration" in rep.markdown and "termination" not in rep.markdown.split("\n\n")[1]


@pytest.mark.asyncio
async def test_outline_and_to_markdown_render_reports() -> None:
    outline = await D.outline(_doc(), title="Outline")
    assert "Limitation of Liability" in outline.markdown and outline.markdown.startswith("# Outline")
    md = await D.findings_to_markdown(
        D.Findings(items=[D.Finding(section="Termination", clause_type="termination", risk="high")])
    )
    assert "| Section |" in md.markdown and "termination" in md.markdown


# --- LLM-backed ops ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_clauses_builds_findings() -> None:
    llm = _StubLLM('[{"section": 0, "clause_type": "liability"}, {"section": 1, "clause_type": "termination"}]')
    out = await D.extract_clauses(_doc(), clause_types=["liability", "termination"], llm=llm)
    assert [(f.section, f.clause_type) for f in out.items] == [
        ("Limitation of Liability", "liability"),
        ("Termination", "termination"),
    ]
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_extract_clauses_rejects_bad_output() -> None:
    with pytest.raises(ValueError, match="not in"):  # clause_type outside the requested set
        await D.extract_clauses(
            _doc(), clause_types=["liability"], llm=_StubLLM('[{"section": 0, "clause_type": "zzz"}]')
        )
    with pytest.raises(ValueError, match="index out of range"):
        await D.extract_clauses(
            _doc(), clause_types=["liability"], llm=_StubLLM('[{"section": 9, "clause_type": "liability"}]')
        )


@pytest.mark.asyncio
async def test_extract_clauses_caps_sections() -> None:
    big = D.Document(sections=[D.Section(heading=str(i), text="x") for i in range(6)])
    with pytest.raises(ValueError, match="exceeds max_sections"):
        await D.extract_clauses(big, clause_types=["a"], max_sections=3, llm=_StubLLM("[]"))


@pytest.mark.asyncio
async def test_classify_risk_fills_risk_and_note() -> None:
    findings = D.Findings(items=[
        D.Finding(section="A", clause_type="liability", text="uncapped"),
        D.Finding(section="B", clause_type="termination", text="mutual"),
    ])
    llm = _StubLLM('[{"risk": "high", "note": "uncapped"}, {"risk": "low", "note": "balanced"}]')
    out = await D.classify_risk(findings, llm=llm)
    assert [f.risk for f in out.items] == ["high", "low"]
    assert out.items[0].note == "uncapped"


@pytest.mark.asyncio
async def test_classify_risk_rejects_bad_output() -> None:
    findings = D.Findings(items=[D.Finding(section="A", clause_type="x"), D.Finding(section="B", clause_type="y")])
    with pytest.raises(ValueError, match="expected 2 results"):
        await D.classify_risk(findings, llm=_StubLLM('[{"risk": "low"}]'))
    with pytest.raises(ValueError, match="risk"):
        await D.classify_risk(findings, llm=_StubLLM('[{"risk": "low"}, {"risk": "extreme"}]'))


@pytest.mark.asyncio
async def test_compare_to_playbook_reads_file_and_flags(tmp_path: Path) -> None:
    pb = tmp_path / "playbook.txt"
    pb.write_text("Payment: Net 30 is standard. Net 45 is a deviation.")
    findings = D.Findings(items=[
        D.Finding(section="Fees", clause_type="payment", text="Net 45"),
        D.Finding(section="Term", clause_type="renewal", text="12 months"),
    ])
    llm = _StubLLM('[{"deviates": true, "note": "Net 45 > Net 30"}, {"deviates": false, "note": "ok"}]')
    out = await D.compare_to_playbook(findings, playbook_path=str(pb), content_fp="", llm=llm)
    assert out.items[0].note.startswith("deviation:") and out.items[1].note.startswith("aligned:")


@pytest.mark.asyncio
async def test_summarize_and_redline_return_prose() -> None:
    rep = await D.summarize_document(_doc(), goal="risk", llm=_StubLLM("A short summary."))
    assert rep.markdown == "A short summary."
    memo = await D.redline(
        D.Findings(items=[D.Finding(section="A", clause_type="liability", risk="high")]),
        llm=_StubLLM("- Cap the liability."),
    )
    assert memo.markdown == "- Cap the liability."


# --- executor / registry contract ----------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_op_needs_a_backend() -> None:
    g = Graph()
    g.add(Node(id="d", op="sample_contract", params={"name": "msa"}))
    g.add(Node(id="e", op="extract_clauses", params={"clause_types": ["termination"]}, inputs={"doc": "d"}))
    with pytest.raises(GraphValidationError, match="needs an LLM"):
        await execute(g, None, llm=None)


def test_llm_ops_hide_the_reserved_param() -> None:
    from vibe.core.graph.operators import get_operator

    for name in ("extract_clauses", "classify_risk", "compare_to_playbook", "summarize_document", "redline"):
        spec = get_operator(name)
        assert spec.needs_llm and "llm" not in spec.param_names and "llm" not in spec.arg_types
