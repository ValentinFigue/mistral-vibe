"""The document-review operator library — the ``doc-reviewer`` agent's toolkit.

This is the second métier (after ``analysis``): a **general document reviewer** whose operators are
**domain-neutral** — the domain (contracts, research papers, policies, RFPs, …) lives in the
*parameters* (the categories, fields, labels, references you pass), not in the operator names. A
:class:`Document` (any file split into :class:`Section` s) flows through the graph, proving the
engine is artifact-agnostic: a ``Document`` renders as ``ref:<fp> Document`` for free (the handle
keys off ``type(result).__name__``; no ``Table`` dispatch).

The shape of a review: **load → clean/segment → locate → extract → classify / compare → report**.
Large documents stay in the content-addressed cache as handles; the agent reasons over a small
terminal report; editing one step re-runs only the dirty subgraph.

Domain specialization is provided by thin **preset blocks** (``contract_review``, ``term_sheet``,
``paper_abstract``, ``gap_check``, and the neutral ``inventory``) that simply supply default
parameters to the neutral ops. The module is deliberately **independent of** ``library.analysis``.

Engine notes: `read_document` reads `.txt`/`.md` natively; `.pdf` via a lazy `pypdf` import (the
optional ``[pdf]`` extra). LLM-backed ops (`@operator(needs_llm=True)`) declare a reserved ``llm``
param the executor injects at run time; they route JSON replies through :func:`_llm_json`, which
retries once on an empty response (reasoning models can spend the whole token budget reasoning) and
then raises an actionable "reduce the input" error rather than failing cryptically. The
deterministic ops run headless.
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
_MAX_EXTRACT_SECTIONS = 80  # cap what one extract call feeds the model
_MAX_ITEMS = 100  # cap per-item LLM ops (classify / compare_to_reference / suggest_edits / write_memo)
_SECTION_EXCERPT = 600  # chars of each section shown to the model
_MAX_REFERENCE_CHARS = 4000
_SAMPLE_DOCS = ("msa", "nda")
_MIN_SECTION_WORDS = 5  # chunk_document merges sections shorter than this into the previous one
_MIN_TERM_USES = 2  # a defined term appearing fewer times than this is only in its own definition
_MAX_NUM_HEADING_LEN = 120  # a numbered line longer than this is prose, not a heading
_CAPS_HEADING_MIN = 2  # shortest ALL-CAPS heading (chars)
_CAPS_HEADING_MAX = 80  # longest ALL-CAPS heading; longer is a shouted paragraph, not a heading

# Default parameter sets for the domain preset blocks (overridable per call).
_CONTRACT_CATEGORIES = (
    "confidentiality", "indemnification", "limitation of liability",
    "termination", "governing law", "fees",
)
_CONTRACT_FIELDS = (
    "parties", "effective_date", "term", "renewal", "governing_law",
    "fees", "notice_period", "liability_cap",
)
_PAPER_FIELDS = ("objective", "method", "data", "results", "limitations")
_RISK_LABELS = ("low", "medium", "high")

# A heading is a markdown ``#`` line, a numbered clause (``1.``, ``2.3)``, ``Section 4 …``), or a
# short ALL-CAPS line — enough to segment plain-text/PDF documents, not just markdown.
_HEADING_NUM = re.compile(r"^\s*(?:section\s+)?\d+(?:\.\d+)*[.)]?\s+\S", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")  # SGML/HTML tags + page markers (EDGAR wrappers etc.)
_QUOTE = r"[\"“”]"  # straight or curly double-quote
# Heuristic defined-term detection: a Capitalized quoted term followed by "means"/"refers to".
_DEFN_RE = re.compile(_QUOTE + r"([A-Z][A-Za-z0-9 /\-]{1,60})" + _QUOTE + r"\s+(?:means|shall mean|will mean|refers to)")
_QUOTED_TERM_RE = re.compile(_QUOTE + r"([A-Z][A-Za-z0-9 /\-]{1,60})" + _QUOTE)
# Heuristic internal cross-reference: "Section/Article/clause/§ 12.3".
_REF_RE = re.compile(r"\b(?:section|article|clause|§)\s+(\d+(?:\.\d+)*)", re.IGNORECASE)
_SECTION_NUM_RE = re.compile(r"^\s*(?:section\s+|article\s+)?(\d+(?:\.\d+)*)", re.IGNORECASE)


class Section(BaseModel):
    """One titled block of a document."""

    heading: str = ""
    text: str = ""


class Document(BaseModel):
    """A loaded, segmented document — the artifact that flows through the review graph."""

    name: str = ""
    sections: list[Section] = Field(default_factory=list)


class Item(BaseModel):
    """One reviewed piece of a document — a located segment or an extracted field.

    ``category`` is its kind (a clause type, a field name, a topic); ``labels`` holds classifier
    outputs keyed by dimension (e.g. ``{"risk": "high"}``); ``note`` is free-text commentary.
    """

    section: str = ""
    category: str = ""
    text: str = ""
    labels: dict[str, str] = Field(default_factory=dict)
    note: str = ""


class ItemSet(BaseModel):
    """The structured result of a review — the ``Document`` métier's analogue of a table."""

    items: list[Item] = Field(default_factory=list)


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


async def _llm_json(
    llm: LLMCaller, prompt: str, *, system: str, op: str, max_tokens: int
) -> list[Any]:
    """Call the model expecting a JSON array, robust to a reasoning model exhausting its budget.

    A reasoning model spends the same ``max_tokens`` budget on reasoning first, so a large input can
    leave *no* visible content (an empty reply). We retry once with double the budget, then fail with
    an actionable error telling the agent to shrink the input — instead of a cryptic "did not return
    a JSON array: ''".
    """
    raw = await llm(prompt, system=system, max_tokens=max_tokens)
    if not raw.strip():
        raw = await llm(prompt, system=system, max_tokens=max_tokens * 2)
    if not raw.strip():
        raise ValueError(
            f"{op}: the model returned no content (empty even at {max_tokens * 2} tokens) — the "
            "input is likely too large for the model's budget. Reduce it: clean_document / "
            "chunk_document / filter_sections first, pass fewer categories/fields/items, or a shorter reference."
        )
    return _parse_json_array(raw, op)


def _numbered_sections(sections: list[Section]) -> str:
    return "\n".join(
        f"[{i}] {s.heading}: {s.text[:_SECTION_EXCERPT]}" for i, s in enumerate(sections)
    )


def _numbered_items(items: list[Item]) -> str:
    return "\n".join(
        f"{i}: [{it.category}] {it.text[:_SECTION_EXCERPT]}" for i, it in enumerate(items)
    )


def _section_number(heading: str) -> str | None:
    """The leading clause number of a heading (``"3.2 Fees"`` → ``"3.2"``), or None."""
    m = _SECTION_NUM_RE.match(heading)
    return m.group(1) if m else None


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
async def sample_document(name: Literal["msa", "nda"]) -> Document:
    """Load a bundled sample document by name (no file path needed). Samples: msa, nda."""
    if name not in _SAMPLE_DOCS:
        raise ValueError(f"unknown sample document {name!r}; available: {list(_SAMPLE_DOCS)}")
    text = (
        importlib.resources.files("vibe.core.graph.library.data") / f"sample_{name}.txt"
    ).read_text()
    return _document(f"sample_{name}", text)


# --- clean / shape / guard -----------------------------------------------------------------


@operator(library=_LIB)
async def clean_document(doc: Document) -> Document:
    """Strip SGML/HTML tags (e.g. EDGAR ``<DOCUMENT>``/``<PAGE>`` wrappers) from headings and text,
    and drop sections that are empty once cleaned — a **pure** cleanup for messy real-world files.
    Run this **before** ``chunk_document`` / ``extract_*`` on documents with markup noise.
    """
    cleaned: list[Section] = []
    for s in doc.sections:
        heading = _TAG_RE.sub("", s.heading).strip()
        text = "\n".join(_TAG_RE.sub("", line) for line in s.text.splitlines()).strip()
        if heading or text:
            cleaned.append(Section(heading=heading, text=text))
    return Document(name=doc.name, sections=cleaned)


@operator(library=_LIB)
async def chunk_document(doc: Document, max_sections: int = _MAX_EXTRACT_SECTIONS) -> Document:
    """Consolidate sections so an ``extract_*`` call fits the model's budget — a **pure** op. Merges
    tiny fragments (fewer than ~5 words, e.g. page-break stubs) into the previous section, then
    coalesces adjacent sections until there are at most ``max_sections``. Run **after**
    ``clean_document`` and before extraction on large documents.
    """
    merged: list[Section] = []
    for s in doc.sections:
        if merged and len(s.text.split()) < _MIN_SECTION_WORDS and not s.text.startswith("#"):
            prev = merged[-1]
            extra = (f"\n{s.heading}" if s.heading else "") + (f"\n{s.text}" if s.text else "")
            merged[-1] = Section(heading=prev.heading, text=(prev.text + extra).strip())
        else:
            merged.append(s)
    if max_sections >= 1 and len(merged) > max_sections:
        group_size = (len(merged) + max_sections - 1) // max_sections
        grouped: list[Section] = []
        for i in range(0, len(merged), group_size):
            chunk = merged[i : i + group_size]
            heading = chunk[0].heading or f"Sections {i + 1}-{i + len(chunk)}"
            text = "\n\n".join(f"{c.heading}\n{c.text}".strip() for c in chunk).strip()
            grouped.append(Section(heading=heading, text=text))
        merged = grouped
    return Document(name=doc.name, sections=merged)


@operator(library=_LIB)
async def filter_sections(doc: Document, contains: str) -> Document:
    """Keep only sections whose heading or text contains ``contains`` (case-insensitive substring)."""
    needle = contains.lower()
    kept = [s for s in doc.sections if needle in s.heading.lower() or needle in s.text.lower()]
    return Document(name=doc.name, sections=kept)


@operator(library=_LIB)
async def search_sections(doc: Document, pattern: str) -> Document:
    """Keep sections whose heading or text matches the regex ``pattern`` (case-insensitive) — a
    **pure** narrowing op, richer than ``filter_sections``. Invalid regex raises a clear error.
    """
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise ValueError(f"search_sections: invalid regex {pattern!r}: {exc}") from exc
    kept = [s for s in doc.sections if rx.search(s.heading) or rx.search(s.text)]
    return Document(name=doc.name, sections=kept)


@operator(library=_LIB)
async def select_sections(doc: Document, headings: list[str]) -> Document:
    """Keep sections whose heading contains any of ``headings`` (case-insensitive) — a **pure** op
    to pick named sections (e.g. ``["termination", "indemnif"]``).
    """
    needles = [h.lower() for h in _as_list(headings)]
    if not needles:
        raise ValueError("select_sections: headings must be non-empty")
    kept = [s for s in doc.sections if any(n in s.heading.lower() for n in needles)]
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
async def extract_segments(
    doc: Document,
    categories: list[str],
    max_sections: int = _MAX_EXTRACT_SECTIONS,
    llm: LLMCaller | None = None,
) -> ItemSet:
    """Locate the section(s) matching each of ``categories`` — one batched LLM call → an ItemSet
    (each item's ``category`` is the matched category, ``text`` the section text). Categories are
    domain-defined: contract clauses (``["indemnification", "termination"]``), paper topics,
    policy requirements, etc. A section may match several categories; unmatched sections are dropped.
    Errors over ``max_sections`` — ``clean_document`` / ``chunk_document`` / ``filter_sections``
    first. Cached per (categories + document).
    """
    assert llm is not None  # guaranteed by the executor for needs_llm ops
    cats = _as_list(categories)
    if not cats:
        raise ValueError("extract_segments: categories must be non-empty")
    sections = doc.sections
    if len(sections) > max_sections:
        raise ValueError(
            f"extract_segments: {len(sections)} sections exceeds max_sections={max_sections}; "
            "clean_document / chunk_document / filter_sections first"
        )
    prompt = (
        f"You are reviewing a document split into numbered sections. For each category in {cats}, "
        f"identify the section(s) about it. Return ONLY a JSON array of objects "
        f'{{"section": <int index>, "category": <one of {cats}>}} — one object per '
        "(section, category) match, no prose. A section may match several categories; omit a "
        f"section that matches none.\n\nSections:\n{_numbered_sections(sections)}"
    )
    entries = await _llm_json(
        llm, prompt, system="You locate document segments. Output only a JSON array.",
        op="extract_segments", max_tokens=4096,
    )
    allowed = set(cats)
    items: list[Item] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"extract_segments: expected objects, got {entry!r}")
        idx, cat = entry.get("section"), entry.get("category")
        if not isinstance(idx, int) or not (0 <= idx < len(sections)):
            raise ValueError(f"extract_segments: section index out of range: {idx!r}")
        if cat not in allowed:
            raise ValueError(f"extract_segments: category {cat!r} not in {cats}")
        s = sections[idx]
        items.append(Item(section=s.heading, category=str(cat), text=s.text[:_SECTION_EXCERPT]))
    return ItemSet(items=items)


@operator(library=_LIB, needs_llm=True)
async def extract_fields(
    doc: Document, fields: list[str], llm: LLMCaller | None = None
) -> ItemSet:
    """Extract named key-value ``fields`` from the document (its "abstract") — one batched LLM call.
    Each found field becomes an item whose ``category`` is the field name and ``text`` is the value.
    Domain-defined fields: contract key terms, paper metadata (method/data/results), résumé fields,
    etc. ``clean_document`` / ``chunk_document`` a large document first. Cached per (fields + document).
    """
    assert llm is not None  # guaranteed by the executor for needs_llm ops
    wanted = _as_list(fields)
    if not wanted:
        raise ValueError("extract_fields: fields must be non-empty")
    if len(doc.sections) > _MAX_EXTRACT_SECTIONS:
        raise ValueError(
            f"extract_fields: {len(doc.sections)} sections exceeds {_MAX_EXTRACT_SECTIONS}; "
            "clean_document / chunk_document / filter_sections first"
        )
    prompt = (
        f"Extract these fields from the document: {wanted}. Return ONLY a JSON array of objects "
        '{"field": <one of the requested fields>, "value": <the value as stated, concise>, '
        '"section": <int section index or null>} — one object per field you find, no prose. '
        "State only what the text says; omit a field that is not present.\n\n"
        f"Sections:\n{_numbered_sections(doc.sections)}"
    )
    entries = await _llm_json(
        llm, prompt, system="You extract document fields. Output only a JSON array.",
        op="extract_fields", max_tokens=4096,
    )
    allowed = set(wanted)
    items: list[Item] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"extract_fields: expected objects, got {entry!r}")
        field = entry.get("field")
        if field not in allowed:
            raise ValueError(f"extract_fields: field {field!r} not in {wanted}")
        idx = entry.get("section")
        heading = (
            doc.sections[idx].heading
            if isinstance(idx, int) and 0 <= idx < len(doc.sections)
            else ""
        )
        items.append(
            Item(section=heading, category=str(field), text=str(entry.get("value") or "").strip())
        )
    return ItemSet(items=items)


@operator(library=_LIB, needs_llm=True)
async def classify(
    items: ItemSet,
    dimension: str,
    labels: list[str],
    max_items: int = _MAX_ITEMS,
    llm: LLMCaller | None = None,
) -> ItemSet:
    """Label each item along a named ``dimension`` with exactly one of ``labels`` (plus a one-line
    reason) — one batched LLM call. General: ``dimension="risk", labels=["low","medium","high"]`` or
    ``dimension="favorability", labels=["favorable","neutral","unfavorable"]`` or sentiment /
    priority / compliance-status. Writes ``item.labels[dimension]`` (stacks across calls); the reason
    goes to ``note`` when empty. Errors over ``max_items``. Cached per (dimension + labels + items).
    """
    assert llm is not None  # guaranteed by the executor for needs_llm ops
    the_items = items.items
    if not the_items:
        return items
    allowed = _as_list(labels)
    if not allowed:
        raise ValueError("classify: labels must be non-empty")
    if len(the_items) > max_items:
        raise ValueError(f"classify: {len(the_items)} items exceeds max_items={max_items}")
    prompt = (
        f"For each item below, assign its {dimension} as exactly one of {allowed} and give a "
        f"one-line reason. Return ONLY a JSON array of {len(the_items)} objects "
        '{"label": <one of the allowed values>, "note": <short reason>}, in order, no prose.\n\n'
        f"Items:\n{_numbered_items(the_items)}"
    )
    parsed = await _llm_json(
        llm, prompt, system=f"You classify document items by {dimension}. Output only a JSON array.",
        op="classify", max_tokens=4096,
    )
    if len(parsed) != len(the_items):
        raise ValueError(f"classify: expected {len(the_items)} results, got {len(parsed)}")
    allowed_set = set(allowed)
    out: list[Item] = []
    for it, res in zip(the_items, parsed, strict=True):
        if not isinstance(res, dict) or res.get("label") not in allowed_set:
            raise ValueError(f"classify: each result needs label ∈ {allowed}, got {res!r}")
        new_labels = {**it.labels, dimension: str(res["label"])}
        out.append(it.model_copy(update={"labels": new_labels, "note": it.note or str(res.get("note", ""))}))
    return ItemSet(items=out)


@operator(library=_LIB, needs_llm=True, reads_file="reference_path")
async def compare_to_reference(
    items: ItemSet,
    reference_path: str,
    criterion: str = "deviation from the reference",
    content_fp: str = "",
    max_items: int = _MAX_ITEMS,
    llm: LLMCaller | None = None,
) -> ItemSet:
    """Compare each item to a **reference document** (a playbook, rubric, spec, style guide, or
    standard) and flag issues per ``criterion`` — one batched LLM call. Supply only ``reference_path``
    (the tool fingerprints it as ``content_fp``). Errors over ``max_items``. Updates each item's
    ``note``. Cached per (items + criterion + reference content).
    """
    assert llm is not None  # guaranteed by the executor for needs_llm ops
    the_items = items.items
    if not the_items:
        return items
    if len(the_items) > max_items:
        raise ValueError(f"compare_to_reference: {len(the_items)} items exceeds max_items={max_items}")
    p = Path(reference_path).expanduser()
    try:
        if p.stat().st_size > _MAX_DOC_BYTES:
            raise ValueError(f"compare_to_reference: reference {reference_path!r} is too large")
        reference = p.read_text()
    except OSError as exc:
        raise ValueError(f"compare_to_reference: cannot read {reference_path!r}: {exc}") from exc
    prompt = (
        f"Compare each item to the reference and decide whether it shows: {criterion}. Return ONLY a "
        f'JSON array of {len(the_items)} objects {{"flag": <true|false>, "note": <short explanation>}}, '
        "in order, no prose.\n\n"
        f"Reference:\n{reference[:_MAX_REFERENCE_CHARS]}\n\nItems:\n{_numbered_items(the_items)}"
    )
    parsed = await _llm_json(
        llm, prompt, system="You compare document items to a reference. Output only a JSON array.",
        op="compare_to_reference", max_tokens=4096,
    )
    if len(parsed) != len(the_items):
        raise ValueError(f"compare_to_reference: expected {len(the_items)} results, got {len(parsed)}")
    out: list[Item] = []
    for it, res in zip(the_items, parsed, strict=True):
        if not isinstance(res, dict) or "flag" not in res:
            raise ValueError(f"compare_to_reference: each result needs a 'flag', got {res!r}")
        prefix = "flagged" if bool(res["flag"]) else "ok"
        out.append(it.model_copy(update={"note": f"{prefix}: {str(res.get('note', '')).strip()}"}))
    return ItemSet(items=out)


@operator(library=_LIB)
async def find_missing(items: ItemSet, required: list[str]) -> Report:
    """Report which ``required`` categories are absent from the items — a **pure** check (no LLM).
    Use after ``extract_segments`` to catch required clauses / topics / checklist items a document
    lacks entirely.
    """
    required = _as_list(required)
    present = {it.category for it in items.items}
    missing = [r for r in required if r not in present]
    body = (
        "\n".join(f"- {m}" for m in missing)
        if missing
        else "_None — every required category is present._"
    )
    return Report(markdown=f"# Missing items\n\n{body}")


@operator(library=_LIB)
async def check_references(doc: Document) -> Report:
    """Flag internal cross-references ("Section 12.3", "Article 5") that don't resolve to a section
    number present in the document — a **pure** consistency check. Heuristic: matches common
    phrasings only and can't distinguish an internal reference from one into an exhibit/other
    document, so treat results as *possible* issues.
    """
    present = {n for n in (_section_number(s.heading) for s in doc.sections) if n}
    referenced: dict[str, str] = {}  # ref number -> first section heading that cites it
    for s in doc.sections:
        for m in _REF_RE.finditer(s.text):
            referenced.setdefault(m.group(1), s.heading or "Preamble")
    unresolved = sorted(num for num in referenced if num not in present)
    if not unresolved:
        body = "_None — every internal reference resolves to a section._"
    else:
        body = "\n".join(f"- **§{num}** (cited in “{referenced[num]}”) — no matching section" for num in unresolved)
    return Report(markdown=f"# Possible unresolved cross-references\n\n{body}")


@operator(library=_LIB)
async def check_definitions(doc: Document) -> Report:
    """Report defined terms that appear to be **unused**, and Capitalized quoted terms that appear
    to be **used but never defined** — a **pure** consistency check. Heuristic: recognizes the
    common ``"Term" means …`` phrasing only, so expect some false positives on unusual drafting.
    """
    full = "\n".join(f"{s.heading}\n{s.text}" for s in doc.sections)
    defined = {m.group(1).strip() for m in _DEFN_RE.finditer(full)}
    quoted = {m.group(1).strip() for m in _QUOTED_TERM_RE.finditer(full)}
    undefined = sorted(quoted - defined)[:50]
    unused = sorted(t for t in defined if full.count(t) < _MIN_TERM_USES)[:50]

    def _block(items: list[str], empty: str) -> str:
        return "\n".join(f"- {t}" for t in items) if items else empty

    md = (
        "# Defined-term check\n\n"
        "## Used but possibly not defined\n" + _block(undefined, "_None found._") + "\n\n"
        "## Defined but possibly not used\n" + _block(unused, "_None found._")
    )
    return Report(markdown=md)


@operator(library=_LIB, needs_llm=True)
async def answer_question(doc: Document, question: str, llm: LLMCaller | None = None) -> Report:
    """Answer a specific ``question`` about the document, **grounded** in its text — one LLM call.
    Quotes the supporting language and replies "Not stated in the document." when the answer is
    absent. Cached per (question + document).
    """
    assert llm is not None  # guaranteed by the executor for needs_llm ops
    if not question.strip():
        raise ValueError("answer_question: question must be non-empty")
    body = "\n\n".join(f"{s.heading}: {s.text[:_SECTION_EXCERPT]}" for s in doc.sections)
    prompt = (
        f"Answer this question about the document, grounded only in its text: {question}\n"
        "Quote the exact supporting language. If the document does not address it, reply exactly "
        '"Not stated in the document." Invent nothing.\n\n'
        f"{body}"
    )
    text = await llm(prompt, system="You are a precise document reviewer.", max_tokens=2048)
    return Report(markdown=text.strip())


@operator(library=_LIB, needs_llm=True)
async def summarize_document(doc: Document, goal: str = "", llm: LLMCaller | None = None) -> Report:
    """Write a short, factual plain-language summary of the document (optionally focused on
    ``goal``) — one LLM call. Cached per (goal + document).
    """
    assert llm is not None  # guaranteed by the executor for needs_llm ops
    focus = f" Focus on: {goal}." if goal else ""
    body = "\n\n".join(f"{s.heading}: {s.text[:_SECTION_EXCERPT]}" for s in doc.sections)
    prompt = (
        f"Summarize this document in 3-5 sentences for a business reader.{focus} Ground every "
        "statement in the text and invent nothing — if something isn't stated, don't claim it.\n\n"
        f"{body}"
    )
    text = await llm(prompt, system="You are a precise document reviewer.", max_tokens=2048)
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
async def items_to_markdown(items: ItemSet, title: str = "Review", max_rows: int = 50) -> Report:
    """Render items as a markdown table (first ``max_rows`` rows) — the clean terminal report.
    The Labels column flattens ``item.labels`` to ``dim=value; …``.
    """

    def cell(v: str) -> str:
        return (v or "").replace("|", "\\|").replace("\n", " ")

    cols = ["Section", "Category", "Labels", "Note", "Excerpt"]
    lines = [f"# {title}", ""]
    if not items.items:
        lines.append("_(no items)_")
        return Report(markdown="\n".join(lines))
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for it in items.items[: max(0, max_rows)]:
        labels = "; ".join(f"{k}={v}" for k, v in it.labels.items())
        lines.append(
            "| " + " | ".join(cell(v) for v in (it.section, it.category, labels, it.note, it.text[:120])) + " |"
        )
    if len(items.items) > max_rows:
        lines += ["", f"_… {len(items.items) - max_rows} more items_"]
    return Report(markdown="\n".join(lines))


@operator(library=_LIB)
async def combine_reports(first: Report, second: Report, title: str = "Review packet") -> Report:
    """Concatenate two reports into one under a title — a **pure** sink for a multi-section
    deliverable. Pipe the first in and wire the second as a kwarg; chain to combine three or more.
    """
    return Report(
        markdown=f"# {title}\n\n{first.markdown.strip()}\n\n---\n\n{second.markdown.strip()}"
    )


@operator(library=_LIB, needs_llm=True)
async def suggest_edits(items: ItemSet, goal: str = "", llm: LLMCaller | None = None) -> Report:
    """Propose concise suggested edits for the flagged/labelled items — one LLM call, a short
    markdown memo. Optionally focused on ``goal``. Run after ``classify`` / ``compare_to_reference``.
    Cached per (goal + items).
    """
    assert llm is not None  # guaranteed by the executor for needs_llm ops
    the_items = items.items
    if not the_items:
        return Report(markdown="_No items to edit._")
    if len(the_items) > _MAX_ITEMS:
        raise ValueError(f"suggest_edits: {len(the_items)} items exceeds max_items={_MAX_ITEMS}")
    focus = f" Focus on: {goal}." if goal else ""
    detail = "\n".join(
        f"{i}: [{it.category}] labels={it.labels or '-'} note={it.note or '-'}\n   {it.text[:_SECTION_EXCERPT]}"
        for i, it in enumerate(the_items)
    )
    prompt = (
        f"You are revising a document.{focus} For each item that is risky, unfavorable, or flagged, "
        "propose a concise suggested edit (one bullet each); omit items that are fine. Return a short "
        "markdown memo, no preamble.\n\n"
        f"Items:\n{detail}"
    )
    text = await llm(prompt, system="You are a precise document reviewer.", max_tokens=2048)
    return Report(markdown=text.strip())


@operator(library=_LIB, needs_llm=True)
async def write_memo(items: ItemSet, goal: str = "", llm: LLMCaller | None = None) -> Report:
    """Write a structured review memo (executive summary → key issues → recommendations) from the
    items — one LLM call. Grounded in the items; optionally focused on ``goal``. Run after
    ``classify`` / ``compare_to_reference``. Cached per (goal + items).
    """
    assert llm is not None  # guaranteed by the executor for needs_llm ops
    the_items = items.items
    if not the_items:
        return Report(markdown="_No items to write up._")
    if len(the_items) > _MAX_ITEMS:
        raise ValueError(f"write_memo: {len(the_items)} items exceeds max_items={_MAX_ITEMS}")
    focus = f" Focus on: {goal}." if goal else ""
    detail = "\n".join(
        f"{i}: [{it.category}] labels={it.labels or '-'} note={it.note or '-'}\n   {it.text[:_SECTION_EXCERPT]}"
        for i, it in enumerate(the_items)
    )
    prompt = (
        f"Write a concise review memo from these items.{focus} Use three markdown sections: "
        "'## Executive summary' (2-3 sentences), '## Key issues' (bullets, most important first), "
        "'## Recommendations' (bullets). Ground every point in the items; invent nothing.\n\n"
        f"Items:\n{detail}"
    )
    text = await llm(prompt, system="You are a precise document reviewer.", max_tokens=2048)
    return Report(markdown=text.strip())


# --- domain preset blocks (thin defaults over the neutral ops) -----------------------------
# NB: no ``reads_file`` operator (read_document / compare_to_reference) appears inside a block —
# content-fingerprint autofill runs on the authored (pre-expansion) graph, so a file-reading op
# must be authored directly.


def _inventory() -> Graph:
    g = Graph()
    g.add(Node(id="e", op="extract_segments", params={"categories": []}))
    g.add(Node(id="r", op="items_to_markdown", params={"title": "Inventory"}, inputs={"items": "e"}))
    return g


def _contract_review() -> Graph:
    g = Graph()
    g.add(Node(id="e", op="extract_segments", params={"categories": list(_CONTRACT_CATEGORIES)}))
    g.add(Node(id="c", op="classify",
               params={"dimension": "risk", "labels": list(_RISK_LABELS)}, inputs={"items": "e"}))
    g.add(Node(id="r", op="items_to_markdown", params={"title": "Contract review"}, inputs={"items": "c"}))
    return g


def _term_sheet() -> Graph:
    g = Graph()
    g.add(Node(id="e", op="extract_fields", params={"fields": list(_CONTRACT_FIELDS)}))
    g.add(Node(id="r", op="items_to_markdown", params={"title": "Term sheet"}, inputs={"items": "e"}))
    return g


def _paper_abstract() -> Graph:
    g = Graph()
    g.add(Node(id="e", op="extract_fields", params={"fields": list(_PAPER_FIELDS)}))
    g.add(Node(id="r", op="items_to_markdown", params={"title": "Paper abstract"}, inputs={"items": "e"}))
    return g


def _gap_check() -> Graph:
    g = Graph()
    g.add(Node(id="e", op="extract_segments", params={"categories": []}))
    g.add(Node(id="m", op="find_missing", params={"required": []}, inputs={"items": "e"}))
    return g


_BLOCKS = [
    BlockDef(
        name="inventory", graph=_inventory(), library=_LIB,
        description="locate the named categories and list them (any document)",
        input_ports={"doc": ("e", "doc")},
        params={"categories": ("e", "categories"), "title": ("r", "title")}, output="r",
    ),
    BlockDef(
        name="contract_review", graph=_contract_review(), library=_LIB,
        description="legal preset: locate contract clauses, then flag their risk (low/medium/high)",
        input_ports={"doc": ("e", "doc")},
        params={"categories": ("e", "categories"), "title": ("r", "title")}, output="r",
    ),
    BlockDef(
        name="term_sheet", graph=_term_sheet(), library=_LIB,
        description="legal preset: extract a contract's key terms as a term sheet",
        input_ports={"doc": ("e", "doc")},
        params={"fields": ("e", "fields"), "title": ("r", "title")}, output="r",
    ),
    BlockDef(
        name="paper_abstract", graph=_paper_abstract(), library=_LIB,
        description="research preset: extract a paper's objective/method/data/results/limitations",
        input_ports={"doc": ("e", "doc")},
        params={"fields": ("e", "fields"), "title": ("r", "title")}, output="r",
    ),
    BlockDef(
        name="gap_check", graph=_gap_check(), library=_LIB,
        description="policy preset: locate categories, then report which required ones are missing",
        input_ports={"doc": ("e", "doc")},
        params={"categories": ("e", "categories"), "required": ("m", "required")}, output="m",
    ),
]


def register() -> None:
    """Register the document-review blocks (idempotent). Operators register on import above."""
    for block in _BLOCKS:
        if not is_block(block.name):
            register_block(block)


register()
