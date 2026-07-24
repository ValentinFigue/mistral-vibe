"""The document-review operator library — the ``doc-reviewer`` agent's toolkit.

This is the *second* métier (after ``analysis``), and its point is to prove the workflow-graph
engine is **artifact-agnostic**: instead of a tabular :class:`Table`, a :class:`Document`
(a contract split into :class:`Section` s) flows through the graph. The handle/excerpt/cache/DSL
machinery is unchanged — a ``Document`` renders as ``ref:<fp> Document`` for free, because the
engine keys the handle off ``type(result).__name__`` and the excerpt off ``model_dump_json()``,
with no ``Table``-specific dispatch.

The shape of a review: **load → segment → extract → classify / compare → report**. Large
documents stay in the content-addressed cache as handles (never re-inlined into the transcript);
the agent reasons over a small terminal report. Editing one step re-runs only the dirty subgraph.

Operators are all tagged ``library="documents"`` for catalog scoping, and every error names the
offending value and lists the valid options — the agent recovers by reading the feedback, exactly
as in the ``analysis`` kit. This module is deliberately **independent of** ``library.analysis``
(no cross-import), so the two métiers share only the engine.

Engine notes: `read_document` reads `.txt`/`.md` natively; PDF is a **lazy** `pypdf` import with an
actionable install hint (install the ``[pdf]`` extra), mirroring the analyst's lazy sklearn. The
extraction/classification operators are **LLM-backed** (`@operator(needs_llm=True)`): they declare
a reserved ``llm`` param the executor injects at run time, so they are live-session only and cache
per (params + input fingerprint). The deterministic ops (`read_document`, `filter_sections`,
`find_missing_clauses`, `outline`, `to_markdown`) run headless — that is what the comparison
harness exercises.
"""

from __future__ import annotations

from collections.abc import Sequence
import importlib.resources
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from vibe.core.graph.blocks import BlockDef, is_block, register_block
from vibe.core.graph.model import Graph, Node
from vibe.core.graph.operators import operator
from vibe.core.llm.types import (
    LLMCaller,  # lightweight (protocol); the `llm` param is executor-injected
)

_LIB = "documents"
_MAX_DOC_BYTES = 10 * 1024 * 1024  # refuse documents bigger than this (guard against OOM / bad PDFs)
_MAX_SECTIONS = 500  # a parse over this many sections is almost certainly a bad split
_MAX_EXTRACT_SECTIONS = 80  # cap what one extract_clauses call feeds the model
_MAX_FINDINGS = 100  # cap per-finding LLM ops (classify_risk / compare_to_playbook / redline)
_SECTION_EXCERPT = 600  # chars of each section shown to the model
_MAX_PLAYBOOK_CHARS = 4000
_RISKS = ("low", "medium", "high")
_SAMPLE_CONTRACTS = ("msa", "nda")
_MAX_NUM_HEADING_LEN = 120  # a numbered line longer than this is prose, not a clause heading
_CAPS_HEADING_MIN = 2  # shortest ALL-CAPS heading (chars)
_CAPS_HEADING_MAX = 80  # longest ALL-CAPS heading; longer is a shouted paragraph, not a heading

# A heading is a markdown ``#`` line, a numbered clause (``1.``, ``2.3)``, ``Section 4 …``), or a
# short ALL-CAPS line — enough to segment plain-text/PDF contracts, not just markdown.
_HEADING_NUM = re.compile(r"^\s*(?:section\s+)?\d+(?:\.\d+)*[.)]?\s+\S", re.IGNORECASE)


class Section(BaseModel):
    """One titled block of a document."""

    heading: str = ""
    text: str = ""


class Document(BaseModel):
    """A loaded, segmented document — the artifact that flows through the review graph."""

    name: str = ""
    sections: list[Section] = Field(default_factory=list)


class Finding(BaseModel):
    """One reviewed clause: where it is, what it is, and (once assessed) its risk and a note."""

    section: str = ""
    clause_type: str = ""
    text: str = ""
    risk: str = ""
    note: str = ""


class Findings(BaseModel):
    """The structured result of a review — the ``Document`` métier's analogue of a table."""

    items: list[Finding] = Field(default_factory=list)


class Report(BaseModel):
    """A rendered markdown report — the clean, small terminal value the model reads."""

    markdown: str


# --- helpers -------------------------------------------------------------------------------


def _as_list(value: Sequence[str] | str | None) -> list[str]:
    """A bare string is one item, not a sequence of characters; None → []."""
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.startswith("#"):
        return True
    if _HEADING_NUM.match(s) and len(s) < _MAX_NUM_HEADING_LEN:
        return True
    if any(c.isalpha() for c in s) and s == s.upper() and _CAPS_HEADING_MIN < len(s) < _CAPS_HEADING_MAX:
        return True
    return False


def _clean_heading(line: str) -> str:
    return line.strip().lstrip("#").strip()


def _split_sections(text: str) -> list[Section]:
    """Segment raw text into sections on headings; text before the first heading is 'Preamble'."""
    sections: list[Section] = []
    heading: str | None = None
    body: list[str] = []

    def flush() -> None:
        joined = "\n".join(body).strip()
        if heading is not None or joined:
            sections.append(Section(heading=(heading or "Preamble"), text=joined))

    for line in text.splitlines():
        if _is_heading(line):
            flush()
            heading = _clean_heading(line)
            body = []
        else:
            body.append(line)
    flush()
    if len(sections) > _MAX_SECTIONS:
        raise ValueError(
            f"read_document: parsed {len(sections)} sections, over the {_MAX_SECTIONS} cap — "
            "the document may be mis-formatted"
        )
    return sections


def _document(name: str, text: str) -> Document:
    return Document(name=name, sections=_split_sections(text))


def _require_pypdf():  # noqa: ANN202 (module, imported lazily)
    try:
        import pypdf
    except ImportError as exc:
        raise ValueError(
            "read_document: reading a PDF needs pypdf — install the extra with "
            "`uv sync --extra pdf` (or `pip install 'vibe[pdf]'`); .txt/.md files need no extra"
        ) from exc
    return pypdf


def _read_pdf(path: Path) -> str:
    pypdf = _require_pypdf()
    reader = pypdf.PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _parse_json_array(raw: str, op: str) -> list[Any]:
    """Extract the JSON array from a model reply, with an actionable error on malformed output."""
    import json

    try:
        arr = json.loads(raw[raw.index("[") : raw.rindex("]") + 1])
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{op}: model did not return a JSON array: {raw[:200]!r}") from exc
    if not isinstance(arr, list):
        raise ValueError(f"{op}: expected a JSON array, got {type(arr).__name__}")
    return arr


def _numbered_sections(sections: list[Section]) -> str:
    return "\n".join(
        f"[{i}] {s.heading}: {s.text[:_SECTION_EXCERPT]}" for i, s in enumerate(sections)
    )


def _numbered_findings(items: list[Finding]) -> str:
    return "\n".join(
        f"{i}: [{it.clause_type}] {it.text[:_SECTION_EXCERPT]}" for i, it in enumerate(items)
    )


# --- load ----------------------------------------------------------------------------------


@operator(library=_LIB, reads_file="path")
async def read_document(path: str, content_fp: str) -> Document:
    """Load a document (.txt/.md, or .pdf with the ``[pdf]`` extra) and segment it into sections.
    Supply only ``path`` — the tool fingerprints the file (``content_fp``) so an edit re-runs the
    dependent subgraph.
    """
    p = Path(path).expanduser()
    try:
        size = p.stat().st_size
    except OSError as exc:
        raise ValueError(f"read_document: cannot read {path!r}: {exc}") from exc
    if size > _MAX_DOC_BYTES:
        raise ValueError(
            f"read_document: {path!r} is {size} bytes, over the {_MAX_DOC_BYTES}-byte cap"
        )
    text = _read_pdf(p) if p.suffix.lower() == ".pdf" else p.read_text()
    return _document(p.name, text)


@operator(library=_LIB)
async def sample_contract(name: Literal["msa", "nda"]) -> Document:
    """Load a bundled sample contract by name (no file path needed). Contracts: msa, nda."""
    if name not in _SAMPLE_CONTRACTS:
        raise ValueError(f"unknown sample contract {name!r}; available: {list(_SAMPLE_CONTRACTS)}")
    text = (
        importlib.resources.files("vibe.core.graph.library.data") / f"sample_{name}.txt"
    ).read_text()
    return _document(f"sample_{name}", text)


# --- shape / guard -------------------------------------------------------------------------


@operator(library=_LIB)
async def filter_sections(doc: Document, contains: str) -> Document:
    """Keep only sections whose heading or text contains ``contains`` (case-insensitive)."""
    needle = contains.lower()
    kept = [s for s in doc.sections if needle in s.heading.lower() or needle in s.text.lower()]
    return Document(name=doc.name, sections=kept)


@operator(library=_LIB)
async def expect_sections(doc: Document, min_count: int) -> Document:
    """Assert the document parsed into at least ``min_count`` sections; pass it through unchanged,
    else fail fast (a parse that yields too few sections usually means a bad split).
    """
    if len(doc.sections) < min_count:
        raise ValueError(
            f"expect_sections: found {len(doc.sections)} section(s), fewer than the expected "
            f"{min_count} — the document may have failed to parse"
        )
    return doc


# --- extract / analyze ---------------------------------------------------------------------


@operator(library=_LIB, needs_llm=True)
async def extract_clauses(
    doc: Document,
    clause_types: list[str],
    max_sections: int = _MAX_EXTRACT_SECTIONS,
    llm: LLMCaller | None = None,
) -> Findings:
    """Find which sections contain each of ``clause_types`` — one batched LLM call → a Findings
    table (section, clause_type, text). A section may match several types; unmatched sections are
    dropped. Errors (rather than truncating) if the document has more than ``max_sections``
    sections — ``filter_sections`` first. Cached per (clause_types + document).
    """
    assert llm is not None  # guaranteed by the executor for needs_llm ops
    types = _as_list(clause_types)
    if not types:
        raise ValueError("extract_clauses: clause_types must be non-empty")
    sections = doc.sections
    if len(sections) > max_sections:
        raise ValueError(
            f"extract_clauses: {len(sections)} sections exceeds max_sections={max_sections}; "
            "filter_sections first"
        )
    prompt = (
        f"You are reviewing a contract split into numbered sections. For each clause type in "
        f"{types}, identify the section(s) containing it. Return ONLY a JSON array of objects "
        f'{{"section": <int index>, "clause_type": <one of {types}>}} — one object per '
        "(section, clause_type) match, no prose. A section may match several types; omit a section "
        f"that matches none.\n\nSections:\n{_numbered_sections(sections)}"
    )
    raw = await llm(
        prompt, system="You extract contract clauses. Output only a JSON array.", max_tokens=4096
    )
    allowed = set(types)
    items: list[Finding] = []
    for entry in _parse_json_array(raw, "extract_clauses"):
        if not isinstance(entry, dict):
            raise ValueError(f"extract_clauses: expected objects, got {entry!r}")
        idx, ctype = entry.get("section"), entry.get("clause_type")
        if not isinstance(idx, int) or not (0 <= idx < len(sections)):
            raise ValueError(f"extract_clauses: section index out of range: {idx!r}")
        if ctype not in allowed:
            raise ValueError(f"extract_clauses: clause_type {ctype!r} not in {types}")
        s = sections[idx]
        items.append(
            Finding(section=s.heading, clause_type=str(ctype), text=s.text[:_SECTION_EXCERPT])
        )
    return Findings(items=items)


@operator(library=_LIB, needs_llm=True)
async def classify_risk(
    findings: Findings, max_findings: int = _MAX_FINDINGS, llm: LLMCaller | None = None
) -> Findings:
    """Assess each finding's risk to the customer as exactly one of low/medium/high with a one-line
    note — one batched LLM call. Errors if there are more than ``max_findings`` findings (extract or
    filter first). Cached per (findings). Returns the findings with ``risk`` and ``note`` filled.
    """
    assert llm is not None  # guaranteed by the executor for needs_llm ops
    items = findings.items
    if not items:
        return findings
    if len(items) > max_findings:
        raise ValueError(
            f"classify_risk: {len(items)} findings exceeds max_findings={max_findings}"
        )
    prompt = (
        f"For each contract clause below, assess its risk to the customer as exactly one of "
        f"{list(_RISKS)} and give a one-line reason. Return ONLY a JSON array of {len(items)} "
        'objects {"risk": <low|medium|high>, "note": <short reason>}, in order, no prose.\n\n'
        f"Clauses:\n{_numbered_findings(items)}"
    )
    raw = await llm(
        prompt, system="You assess contract risk. Output only a JSON array.", max_tokens=4096
    )
    parsed = _parse_json_array(raw, "classify_risk")
    if len(parsed) != len(items):
        raise ValueError(f"classify_risk: expected {len(items)} results, got {len(parsed)}")
    out: list[Finding] = []
    for it, res in zip(items, parsed, strict=True):
        if not isinstance(res, dict) or res.get("risk") not in _RISKS:
            raise ValueError(
                f"classify_risk: each result needs risk ∈ {list(_RISKS)}, got {res!r}"
            )
        out.append(it.model_copy(update={"risk": str(res["risk"]), "note": str(res.get("note", ""))}))
    return Findings(items=out)


@operator(library=_LIB, needs_llm=True, reads_file="playbook_path")
async def compare_to_playbook(
    findings: Findings,
    playbook_path: str,
    content_fp: str = "",
    max_findings: int = _MAX_FINDINGS,
    llm: LLMCaller | None = None,
) -> Findings:
    """Compare each finding to a standard-terms playbook file and flag deviations — one batched LLM
    call. Supply only ``playbook_path`` (the tool fingerprints it as ``content_fp``). Errors over
    ``max_findings``. Cached per (findings + playbook content). Updates each finding's ``note``.
    """
    assert llm is not None  # guaranteed by the executor for needs_llm ops
    items = findings.items
    if not items:
        return findings
    if len(items) > max_findings:
        raise ValueError(
            f"compare_to_playbook: {len(items)} findings exceeds max_findings={max_findings}"
        )
    p = Path(playbook_path).expanduser()
    try:
        if p.stat().st_size > _MAX_DOC_BYTES:
            raise ValueError(f"compare_to_playbook: playbook {playbook_path!r} is too large")
        playbook = p.read_text()
    except OSError as exc:
        raise ValueError(f"compare_to_playbook: cannot read {playbook_path!r}: {exc}") from exc
    prompt = (
        "Compare each contract clause to the standard playbook and decide whether it deviates from "
        f"the preferred position. Return ONLY a JSON array of {len(items)} objects "
        '{"deviates": <true|false>, "note": <short explanation>}, in order, no prose.\n\n'
        f"Playbook:\n{playbook[:_MAX_PLAYBOOK_CHARS]}\n\nClauses:\n{_numbered_findings(items)}"
    )
    raw = await llm(
        prompt,
        system="You compare contract clauses to a playbook. Output only a JSON array.",
        max_tokens=4096,
    )
    parsed = _parse_json_array(raw, "compare_to_playbook")
    if len(parsed) != len(items):
        raise ValueError(f"compare_to_playbook: expected {len(items)} results, got {len(parsed)}")
    out: list[Finding] = []
    for it, res in zip(items, parsed, strict=True):
        if not isinstance(res, dict) or "deviates" not in res:
            raise ValueError(f"compare_to_playbook: each result needs a 'deviates' flag, got {res!r}")
        deviates = bool(res["deviates"])
        prefix = "deviation" if deviates else "aligned"
        out.append(it.model_copy(update={"note": f"{prefix}: {str(res.get('note', '')).strip()}"}))
    return Findings(items=out)


@operator(library=_LIB)
async def find_missing_clauses(findings: Findings, required: list[str]) -> Report:
    """Report which ``required`` clause types are absent from the findings — a **pure** check (no
    LLM). Use after ``extract_clauses`` to catch clauses a contract is missing entirely.
    """
    required = _as_list(required)
    present = {it.clause_type for it in findings.items}
    missing = [r for r in required if r not in present]
    body = (
        "\n".join(f"- {m}" for m in missing)
        if missing
        else "_None — every required clause is present._"
    )
    return Report(markdown=f"# Missing clauses\n\n{body}")


@operator(library=_LIB, needs_llm=True)
async def summarize_document(doc: Document, goal: str = "", llm: LLMCaller | None = None) -> Report:
    """Write a short, factual plain-language summary of the document (optionally focused on
    ``goal``) — one LLM call. Cached per (goal + document).
    """
    assert llm is not None  # guaranteed by the executor for needs_llm ops
    focus = f" Focus on: {goal}." if goal else ""
    body = "\n\n".join(f"{s.heading}: {s.text[:_SECTION_EXCERPT]}" for s in doc.sections)
    prompt = (
        f"Summarize this contract in 3-5 sentences for a business reader.{focus} Ground every "
        "statement in the text and invent nothing — if a term isn't stated, don't claim it.\n\n"
        f"{body}"
    )
    text = await llm(prompt, system="You are a precise contract reviewer.", max_tokens=2048)
    return Report(markdown=text.strip())


# --- report --------------------------------------------------------------------------------


@operator(library=_LIB)
async def outline(doc: Document, title: str = "Document outline") -> Report:
    """Render the document's structure (headings + word counts) as a markdown report — a **pure**
    sink so a document pipeline can end in a small, clean terminal value without an LLM call.
    """
    lines = [f"# {title}", "", f"**{doc.name}** — {len(doc.sections)} section(s)", ""]
    lines += [f"- **{s.heading}** ({len(s.text.split())} words)" for s in doc.sections]
    return Report(markdown="\n".join(lines))


@operator(library=_LIB)
async def findings_to_markdown(findings: Findings, title: str = "Review", max_rows: int = 50) -> Report:
    """Render findings as a markdown table (first ``max_rows`` rows) — the clean terminal report."""

    def cell(v: str) -> str:
        return (v or "").replace("|", "\\|").replace("\n", " ")

    cols = ["Section", "Clause", "Risk", "Note", "Excerpt"]
    lines = [f"# {title}", ""]
    if not findings.items:
        lines.append("_(no findings)_")
        return Report(markdown="\n".join(lines))
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for it in findings.items[: max(0, max_rows)]:
        lines.append(
            "| "
            + " | ".join(
                cell(v)
                for v in (it.section, it.clause_type, it.risk, it.note, it.text[:120])
            )
            + " |"
        )
    if len(findings.items) > max_rows:
        lines += ["", f"_… {len(findings.items) - max_rows} more findings_"]
    return Report(markdown="\n".join(lines))


@operator(library=_LIB, needs_llm=True)
async def redline(findings: Findings, llm: LLMCaller | None = None) -> Report:
    """Propose concise redlines (suggested edits) for the risky/deviating clauses — one LLM call,
    a short markdown memo. Cached per (findings). Run after ``classify_risk`` / ``compare_to_playbook``.
    """
    assert llm is not None  # guaranteed by the executor for needs_llm ops
    items = findings.items
    if not items:
        return Report(markdown="_No findings to redline._")
    if len(items) > _MAX_FINDINGS:
        raise ValueError(f"redline: {len(items)} findings exceeds max_findings={_MAX_FINDINGS}")
    detail = "\n".join(
        f"{i}: [{it.clause_type}] risk={it.risk or '?'} note={it.note or '-'}\n   {it.text[:_SECTION_EXCERPT]}"
        for i, it in enumerate(items)
    )
    prompt = (
        "You are redlining a contract. For each clause that is risky or deviates from standard "
        "terms, propose a concise suggested edit (one bullet each); omit clauses that are fine. "
        "Return a short markdown memo, no preamble.\n\n"
        f"Clauses:\n{detail}"
    )
    text = await llm(prompt, system="You are a precise contract reviewer.", max_tokens=2048)
    return Report(markdown=text.strip())


# --- classic-review blocks (the workflows, encoded) ----------------------------------------
# NB: no ``reads_file`` operator appears inside a block — content-fingerprint autofill runs on the
# authored (pre-expansion) graph, so a file-reading op must be authored directly (as in analysis).


def _risk_review() -> Graph:
    g = Graph()
    g.add(Node(id="e", op="extract_clauses", params={"clause_types": []}))
    g.add(Node(id="c", op="classify_risk", inputs={"findings": "e"}))
    g.add(Node(id="r", op="findings_to_markdown", params={"title": "Risk review"}, inputs={"findings": "c"}))
    return g


def _clause_inventory() -> Graph:
    g = Graph()
    g.add(Node(id="e", op="extract_clauses", params={"clause_types": []}))
    g.add(Node(id="r", op="findings_to_markdown", params={"title": "Clause inventory"}, inputs={"findings": "e"}))
    return g


def _missing_clauses() -> Graph:
    g = Graph()
    g.add(Node(id="e", op="extract_clauses", params={"clause_types": []}))
    g.add(Node(id="m", op="find_missing_clauses", params={"required": []}, inputs={"findings": "e"}))
    return g


_BLOCKS = [
    BlockDef(
        name="risk_review", graph=_risk_review(), library=_LIB,
        description="extract clauses, then flag their risk — a full risk pass over a contract",
        input_ports={"doc": ("e", "doc")},
        params={"clause_types": ("e", "clause_types"), "title": ("r", "title")}, output="r",
    ),
    BlockDef(
        name="clause_inventory", graph=_clause_inventory(), library=_LIB,
        description="which of the named clause types appear in this contract",
        input_ports={"doc": ("e", "doc")},
        params={"clause_types": ("e", "clause_types"), "title": ("r", "title")}, output="r",
    ),
    BlockDef(
        name="missing_clauses", graph=_missing_clauses(), library=_LIB,
        description="which required clause types are absent from the contract",
        input_ports={"doc": ("e", "doc")},
        params={"clause_types": ("e", "clause_types"), "required": ("m", "required")}, output="m",
    ),
]


def register() -> None:
    """Register the document-review blocks (idempotent). Operators register on import above."""
    for block in _BLOCKS:
        if not is_block(block.name):
            register_block(block)


register()
