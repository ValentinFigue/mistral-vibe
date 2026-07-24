"""The teaching operator library — the ``teacher`` agent's toolkit (the *authoring* half of the
Teaching métier).

This is the third métier (after ``analysis`` and ``documents``): a **lesson author** whose operators
are **domain-neutral** — the domain (subject, grade band, standard, reading level) lives in the
*parameters* you pass, not in the operator names. The same ``generate_items`` builds a reading-comprehension
worksheet, a math quiz, or a science formative check; you choose the standard, the objectives, and the
item kinds.

The shape of authoring a unit: **frame → generate → adapt → assess → bound**, mirroring the pipeline
shape of the other métiers (load → … → report). A :class:`Standard` anchors the graph; an
:class:`ObjectiveSet`, a :class:`Passage`, and an :class:`ItemSet` flow through it; the terminal value
is a small :class:`Report` the teacher reads — or, for the **bound** steps, a :class:`TutorContract`.

The **join** to the (future) student tutor: the *bound* operators (`set_reveal_policy`,
`define_hint_ladder`, `set_escalation_rules`, `set_done_criteria`) chain into a first-class,
content-addressed :class:`TutorContract` artifact. That contract *is* the boundary a student-facing
tutor would run inside — the teacher's authored graph provably bounds the tutor. Note the deliberate
absence of an ``answer`` operator on the contract's move set: authoring produces answer keys for
*teachers*; a tutor must not simply hand a student the answer, so ``answer`` is named in the contract's
``forbidden_moves``.

Domain specialization is provided by thin **preset blocks** (`reading_lesson`, `quiz_from_standard`,
`differentiated_worksheet`, `tutor_contract`) that simply supply default parameters to the neutral ops
— including a one-click contract, so the boundary comes almost for free. The module is deliberately
**independent of** ``library.analysis`` and ``library.doc_reviewer``.

Engine notes: LLM-backed ops (`@operator(needs_llm=True)`) declare a reserved ``llm`` param the executor
injects at run time; the structured ones route JSON replies through :func:`_llm_json`, which retries once
on an empty response (reasoning models can spend the whole token budget reasoning) and then raises an
actionable "reduce the request" error rather than failing cryptically. The pure ops (renders, coverage
checks, and every *bound* op) run headless.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from vibe.core.graph.blocks import BlockDef, is_block, register_block
from vibe.core.graph.model import Graph, Node
from vibe.core.graph.operators import operator
from vibe.core.llm.types import (
    LLMCaller,  # lightweight (protocol); the `llm` param is executor-injected
)

_LIB = "teaching"
_MAX_OBJECTIVES = 20  # cap objectives generated / carried through an LLM op
_MAX_ITEMS = 100  # cap per-item LLM ops (generate_items count, differentiate, add_scaffolds, …)
_MAX_ITEM_COUNT = 40  # cap a single generate_items request
_TEXT_EXCERPT = 800  # chars of a passage/objective shown to the model
_ITEM_EXCERPT = 400  # chars of an item stem shown to the model
_KINDS = ("mcq", "short", "essay")  # item kinds a worksheet/quiz may contain
_DIFFICULTIES = ("easy", "medium", "hard")
_MIN_READING_GRADE, _MAX_READING_GRADE = 1, 13  # grade band a passage may target
_MIN_RUBRIC_LEVELS, _MAX_RUBRIC_LEVELS = 2, 6  # performance levels per rubric criterion

# The moves a bounded tutor may make (the contract's default `allowed_moves`). There is deliberately
# no `answer` here — revealing the full solution is governed by the contract's reveal policy, and
# `answer` is named in `forbidden_moves`.
_DEFAULT_MOVES = (
    "hint", "scaffold", "worked_example", "check_understanding",
    "socratic_prompt", "encourage", "escalate_to_teacher",
)

# Bundled sample standards so the métier runs with no external files (mirrors `sample_document`).
_SAMPLE_STANDARDS: dict[str, dict[str, str]] = {
    "ccss_ela_5": {
        "code": "CCSS.ELA-LITERACY.RI.5.2",
        "subject": "English Language Arts",
        "grade": "5",
        "description": (
            "Determine two or more main ideas of a text and explain how they are supported by key "
            "details; summarize the text."
        ),
    },
    "ccss_math_6": {
        "code": "CCSS.MATH.CONTENT.6.RP.A.3",
        "subject": "Mathematics",
        "grade": "6",
        "description": (
            "Use ratio and rate reasoning to solve real-world and mathematical problems, e.g. by "
            "reasoning about tables of equivalent ratios, tape diagrams, double number lines, or equations."
        ),
    },
    "ngss_ms_ls2": {
        "code": "MS-LS2-3",
        "subject": "Science",
        "grade": "7",
        "description": (
            "Develop a model to describe the cycling of matter and flow of energy among living and "
            "nonliving parts of an ecosystem."
        ),
    },
}


class Standard(BaseModel):
    """A learning standard — the anchor every artifact in a unit traces back to."""

    code: str = ""
    subject: str = ""
    grade: str = ""
    description: str = ""


class Objective(BaseModel):
    """One measurable learning objective aligned to a standard."""

    text: str = ""
    standard_code: str = ""
    bloom_level: str = ""  # remember / understand / apply / analyze / evaluate / create


class ObjectiveSet(BaseModel):
    """A set of objectives for a unit — flows through the authoring graph."""

    standard_code: str = ""
    items: list[Objective] = Field(default_factory=list)


class Passage(BaseModel):
    """A generated reading passage (or adapted/translated variant) — a flowing artifact."""

    title: str = ""
    text: str = ""
    reading_grade: int = 0
    word_count: int = 0


class Item(BaseModel):
    """One assessment item — the teaching métier's analogue of a reviewed piece.

    ``kind`` is the item type; ``choices`` are the options for an ``mcq``; ``answer`` is the answer-key
    value (for teachers); ``objective`` is the objective it serves (provenance); ``scaffolds`` are
    supports added for differentiation; ``labels`` holds any classifier outputs keyed by dimension.
    """

    stem: str = ""
    kind: Literal["mcq", "short", "essay"] = "short"
    choices: list[str] = Field(default_factory=list)
    answer: str = ""
    objective: str = ""
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    scaffolds: list[str] = Field(default_factory=list)
    labels: dict[str, str] = Field(default_factory=dict)


class ItemSet(BaseModel):
    """A set of assessment items — a worksheet or a quiz before it is rendered."""

    items: list[Item] = Field(default_factory=list)


class RubricCriterion(BaseModel):
    """One row of a rubric: a named criterion with an ordered list of performance levels."""

    name: str = ""
    levels: list[str] = Field(default_factory=list)


class Rubric(BaseModel):
    """A scoring rubric — an ordered set of criteria, each with performance levels."""

    title: str = ""
    criteria: list[RubricCriterion] = Field(default_factory=list)


class HintRung(BaseModel):
    """One rung of a tutor's hint ladder, ordered least → most revealing."""

    order: int = 0
    text: str = ""


class TutorContract(BaseModel):
    """The boundary a student-facing tutor runs inside — authored by the teacher's *bound* steps.

    A content-addressed artifact like any other: a tutor session would load it by fingerprint and be
    provably constrained by it. ``forbidden_moves`` names ``answer`` explicitly — a tutor may never
    simply hand over the solution; whether a full solution is *ever* surfaced is governed by
    ``reveal_policy``.
    """

    objectives: list[str] = Field(default_factory=list)  # what the student works toward
    item_scope: list[str] = Field(default_factory=list)  # item stems the tutor may help with
    reveal_policy: Literal["never", "on_request", "after_attempts"] = "after_attempts"
    reveal_after_attempts: int = 3
    hint_ladder: list[HintRung] = Field(default_factory=list)  # ordered, least → most revealing
    escalation_triggers: list[str] = Field(default_factory=list)
    done_criteria: str = ""
    allowed_moves: list[str] = Field(default_factory=lambda: list(_DEFAULT_MOVES))
    forbidden_moves: list[str] = Field(default_factory=lambda: ["answer"])


class Report(BaseModel):
    """A rendered markdown report — the clean, small terminal value the teacher reads."""

    markdown: str


# --- helpers -------------------------------------------------------------------------------


def _as_list(value: Sequence[str] | str | None) -> list[str]:
    """A bare string is one item, not a sequence of characters; None → []."""
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _parse_json_array(raw: str, op: str) -> list[Any]:
    """Extract the JSON array from a model reply, with an actionable error on malformed output."""
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

    A reasoning model spends the same ``max_tokens`` budget on reasoning first, so a large request can
    leave *no* visible content (an empty reply). We retry once with double the budget, then fail with
    an actionable error telling the agent to shrink the request — instead of a cryptic "did not return
    a JSON array: ''".
    """
    raw = await llm(prompt, system=system, max_tokens=max_tokens)
    if not raw.strip():
        raw = await llm(prompt, system=system, max_tokens=max_tokens * 2)
    if not raw.strip():
        raise ValueError(
            f"{op}: the model returned no content (empty even at {max_tokens * 2} tokens) — the "
            "request is likely too large for the model's budget. Reduce it: pass fewer objectives / "
            "items / a smaller count, or a shorter passage."
        )
    return _parse_json_array(raw, op)


def _objectives_block(objs: list[Objective]) -> str:
    return "\n".join(f"[{i}] {o.text}" for i, o in enumerate(objs))


def _numbered_items(items: list[Item]) -> str:
    return "\n".join(
        f"{i}: ({it.kind}) {it.stem[:_ITEM_EXCERPT]}" for i, it in enumerate(items)
    )


def _cell(v: str) -> str:
    return (v or "").replace("|", "\\|").replace("\n", " ")


# --- frame -----------------------------------------------------------------------------------


@operator(library=_LIB)
async def set_standard(code: str, subject: str = "", grade: str = "", description: str = "") -> Standard:
    """Anchor a unit to a learning standard — a **pure** op. ``code`` is the standard's identifier
    (e.g. ``"CCSS.ELA-LITERACY.RI.5.2"``); the other fields are optional context. Every downstream
    artifact traces back to this standard.
    """
    if not code.strip():
        raise ValueError("set_standard: code must be non-empty")
    return Standard(code=code.strip(), subject=subject.strip(), grade=str(grade).strip(),
                    description=description.strip())


@operator(library=_LIB)
async def sample_standard(name: Literal["ccss_ela_5", "ccss_math_6", "ngss_ms_ls2"]) -> Standard:
    """Load a bundled sample standard by name (no file needed). Samples: ccss_ela_5, ccss_math_6,
    ngss_ms_ls2 — for trying a pipeline end-to-end without authoring a standard first.
    """
    if name not in _SAMPLE_STANDARDS:
        raise ValueError(f"unknown sample standard {name!r}; available: {list(_SAMPLE_STANDARDS)}")
    return Standard(**_SAMPLE_STANDARDS[name])


@operator(library=_LIB, needs_llm=True)
async def set_objectives(
    standard: Standard, count: int = 4, focus: str = "", llm: LLMCaller | None = None
) -> ObjectiveSet:
    """Draft ``count`` measurable learning objectives aligned to the ``standard`` — one batched LLM
    call. Each objective is tagged with the standard code and a Bloom's level. Optionally narrow with
    ``focus`` (e.g. ``"vocabulary in context"``). Cached per (standard + count + focus).
    """
    assert llm is not None  # guaranteed by the executor for needs_llm ops
    if count < 1 or count > _MAX_OBJECTIVES:
        raise ValueError(f"set_objectives: count must be between 1 and {_MAX_OBJECTIVES}")
    focus_line = f" Focus on: {focus}." if focus else ""
    prompt = (
        f"Write {count} measurable, student-facing learning objectives for this standard.{focus_line} "
        "Each should start with an observable verb. Return ONLY a JSON array of objects "
        '{"text": <the objective>, "bloom_level": <one of remember|understand|apply|analyze|evaluate|create>}, '
        "no prose.\n\n"
        f"Standard {standard.code} ({standard.subject}, grade {standard.grade}): {standard.description}"
    )
    entries = await _llm_json(
        llm, prompt, system="You are an instructional designer. Output only a JSON array.",
        op="set_objectives", max_tokens=2048,
    )
    objs: list[Objective] = []
    for entry in entries[:count]:
        if not isinstance(entry, dict) or not str(entry.get("text", "")).strip():
            raise ValueError(f"set_objectives: each entry needs a 'text', got {entry!r}")
        objs.append(Objective(
            text=str(entry["text"]).strip(),
            standard_code=standard.code,
            bloom_level=str(entry.get("bloom_level", "")).strip(),
        ))
    return ObjectiveSet(standard_code=standard.code, items=objs)


@operator(library=_LIB, needs_llm=True)
async def sequence_lesson(objectives: ObjectiveSet, minutes: int = 45, llm: LLMCaller | None = None) -> Report:
    """Draft a lesson sequence (a timed set of activities) that covers the ``objectives`` in
    ``minutes`` — one LLM call returning a markdown plan. Cached per (objectives + minutes).
    """
    assert llm is not None  # guaranteed by the executor for needs_llm ops
    if not objectives.items:
        raise ValueError("sequence_lesson: objectives must be non-empty")
    prompt = (
        f"Draft a {minutes}-minute lesson sequence covering these objectives. Use markdown: a short "
        "'## Overview', then a '## Sequence' with timed steps (warm-up → instruction → guided practice "
        "→ independent practice → check for understanding), then '## Materials'. Be concrete and "
        "classroom-ready; invent nothing beyond the objectives.\n\n"
        f"Objectives:\n{_objectives_block(objectives.items)}"
    )
    text = await llm(prompt, system="You are an experienced teacher and lesson planner.", max_tokens=2048)
    return Report(markdown=text.strip())


# --- generate --------------------------------------------------------------------------------


@operator(library=_LIB, needs_llm=True)
async def generate_passage(
    objectives: ObjectiveSet, reading_grade: int, words: int = 400, topic: str = "",
    llm: LLMCaller | None = None,
) -> Passage:
    """Generate an original reading passage at a target ``reading_grade``, aligned to the
    ``objectives`` — one LLM call. Optionally steer with ``topic``. Returns a :class:`Passage`; the
    ``word_count`` is measured from the generated text. Cached per (objectives + reading_grade + words + topic).
    """
    assert llm is not None  # guaranteed by the executor for needs_llm ops
    if not objectives.items:
        raise ValueError("generate_passage: objectives must be non-empty")
    if reading_grade < _MIN_READING_GRADE or reading_grade > _MAX_READING_GRADE:
        raise ValueError(
            f"generate_passage: reading_grade must be between {_MIN_READING_GRADE} and {_MAX_READING_GRADE}"
        )
    topic_line = f" Topic: {topic}." if topic else ""
    prompt = (
        f"Write an original, engaging informational passage of about {words} words at a grade-"
        f"{reading_grade} reading level, suitable for teaching these objectives.{topic_line} Begin with "
        "a title line, then the passage. Keep it factually sound and self-contained.\n\n"
        f"Objectives:\n{_objectives_block(objectives.items)}"
    )
    text = (await llm(prompt, system="You are a curriculum writer.", max_tokens=2048)).strip()
    lines = text.splitlines()
    title = (lines[0].lstrip("# ").strip() if lines else "") or (topic or "Reading passage")
    body = "\n".join(lines[1:]).strip() if len(lines) > 1 else text
    return Passage(title=title, text=body, reading_grade=reading_grade, word_count=len(body.split()))


@operator(library=_LIB, needs_llm=True)
async def generate_items(
    source: Passage | ObjectiveSet,
    count: int = 6,
    kinds: list[str] | None = None,
    llm: LLMCaller | None = None,
) -> ItemSet:
    """Generate ``count`` assessment items from a ``source`` (a :class:`Passage` for reading questions,
    or an :class:`ObjectiveSet` for standalone questions) — one batched LLM call. ``kinds`` restricts
    the item types (any of ``mcq``, ``short``, ``essay``; default ``["mcq"]``). Each item carries its
    answer (for the teacher's key) and the objective it serves. Cached per (source + count + kinds).
    """
    assert llm is not None  # guaranteed by the executor for needs_llm ops
    if count < 1 or count > _MAX_ITEM_COUNT:
        raise ValueError(f"generate_items: count must be between 1 and {_MAX_ITEM_COUNT}")
    wanted = _as_list(kinds) or ["mcq"]
    bad = [k for k in wanted if k not in _KINDS]
    if bad:
        raise ValueError(f"generate_items: unknown item kind(s) {bad}; allowed: {list(_KINDS)}")
    if isinstance(source, Passage):
        context = f"Passage titled {source.title!r} (grade {source.reading_grade}):\n{source.text[:_TEXT_EXCERPT * 3]}"
    else:
        context = f"Objectives:\n{_objectives_block(source.items)}"
    prompt = (
        f"Write {count} assessment items based on the material below. Use only these kinds: {wanted}. "
        "Return ONLY a JSON array of objects "
        '{"stem": <the question>, "kind": <one of the allowed kinds>, "choices": [<options, for mcq only>], '
        '"answer": <the answer, or a model answer>, "objective": <the objective it assesses>, '
        '"difficulty": <one of easy|medium|hard>}, no prose. Give an mcq exactly 4 plausible choices.\n\n'
        f"{context}"
    )
    entries = await _llm_json(
        llm, prompt, system="You write standards-aligned assessment items. Output only a JSON array.",
        op="generate_items", max_tokens=4096,
    )
    allowed = set(wanted)
    items: list[Item] = []
    for entry in entries[:count]:
        if not isinstance(entry, dict) or not str(entry.get("stem", "")).strip():
            raise ValueError(f"generate_items: each entry needs a 'stem', got {entry!r}")
        kind = str(entry.get("kind", wanted[0]))
        if kind not in allowed:
            raise ValueError(f"generate_items: item kind {kind!r} not in {wanted}")
        difficulty = str(entry.get("difficulty", "medium"))
        if difficulty not in _DIFFICULTIES:
            difficulty = "medium"
        items.append(Item(
            stem=str(entry["stem"]).strip(),
            kind=kind,  # type: ignore[arg-type]
            choices=[str(c) for c in (entry.get("choices") or [])] if kind == "mcq" else [],
            answer=str(entry.get("answer", "")).strip(),
            objective=str(entry.get("objective", "")).strip(),
            difficulty=difficulty,  # type: ignore[arg-type]
        ))
    return ItemSet(items=items)


@operator(library=_LIB)
async def build_worksheet(items: ItemSet, title: str = "Worksheet") -> Report:
    """Render items as a student-facing **worksheet** (no answers) — a **pure** sink. Shows each
    item's stem, mcq choices, and any scaffolds; answers are omitted (use ``make_answer_key`` for those).
    """
    lines = [f"# {title}", ""]
    if not items.items:
        lines.append("_(no items)_")
        return Report(markdown="\n".join(lines))
    for i, it in enumerate(items.items, 1):
        lines.append(f"**{i}.** {it.stem}")
        if it.kind == "mcq" and it.choices:
            lines += [f"   - {chr(65 + j)}. {c}" for j, c in enumerate(it.choices)]
        for s in it.scaffolds:
            lines.append(f"   > _hint: {s}_")
        lines.append("")
    return Report(markdown="\n".join(lines).strip())


@operator(library=_LIB)
async def build_quiz(items: ItemSet, title: str = "Quiz") -> Report:
    """Render items as a student-facing **quiz** (no answers) — a **pure** sink. Like ``build_worksheet``
    but tags each item with its kind and difficulty for a graded assessment.
    """
    lines = [f"# {title}", ""]
    if not items.items:
        lines.append("_(no items)_")
        return Report(markdown="\n".join(lines))
    for i, it in enumerate(items.items, 1):
        lines.append(f"**{i}.** _({it.kind}, {it.difficulty})_ {it.stem}")
        if it.kind == "mcq" and it.choices:
            lines += [f"   - {chr(65 + j)}. {c}" for j, c in enumerate(it.choices)]
        lines.append("")
    return Report(markdown="\n".join(lines).strip())


# --- adapt -----------------------------------------------------------------------------------


@operator(library=_LIB, needs_llm=True)
async def adapt_reading_level(passage: Passage, reading_grade: int, llm: LLMCaller | None = None) -> Passage:
    """Rewrite a ``passage`` to a different ``reading_grade`` while preserving its meaning — one LLM
    call. Returns a new :class:`Passage` with the target grade and a fresh word count. Cached per
    (passage + reading_grade).
    """
    assert llm is not None  # guaranteed by the executor for needs_llm ops
    if reading_grade < _MIN_READING_GRADE or reading_grade > _MAX_READING_GRADE:
        raise ValueError(
            f"adapt_reading_level: reading_grade must be between {_MIN_READING_GRADE} and {_MAX_READING_GRADE}"
        )
    prompt = (
        f"Rewrite this passage at a grade-{reading_grade} reading level. Preserve every fact and the "
        "overall meaning; change only vocabulary and sentence complexity. Return only the rewritten "
        f"passage, no preamble.\n\nTitle: {passage.title}\n\n{passage.text}"
    )
    text = (await llm(prompt, system="You are a curriculum writer.", max_tokens=2048)).strip()
    return Passage(title=passage.title, text=text, reading_grade=reading_grade, word_count=len(text.split()))


@operator(library=_LIB, needs_llm=True)
async def differentiate(
    items: ItemSet, level: Literal["support", "core", "stretch"], llm: LLMCaller | None = None
) -> ItemSet:
    """Rewrite each item for a differentiation ``level`` — ``support`` (more accessible), ``core``
    (grade level), or ``stretch`` (extension) — one batched LLM call. Preserves each item's kind and
    objective; updates the stem, choices, answer, and difficulty. Errors over ``max_items``. Cached
    per (items + level).
    """
    assert llm is not None  # guaranteed by the executor for needs_llm ops
    the_items = items.items
    if not the_items:
        return items
    if len(the_items) > _MAX_ITEMS:
        raise ValueError(f"differentiate: {len(the_items)} items exceeds max_items={_MAX_ITEMS}")
    prompt = (
        f"Rewrite each item below for a '{level}' learner (support = more accessible, core = grade "
        "level, stretch = extension). Keep the same kind and what it assesses. Return ONLY a JSON array "
        f"of {len(the_items)} objects "
        '{"stem": <rewritten stem>, "choices": [<for mcq>], "answer": <answer>, '
        '"difficulty": <one of easy|medium|hard>}, in order, no prose.\n\n'
        f"Items:\n{_numbered_items(the_items)}"
    )
    parsed = await _llm_json(
        llm, prompt, system="You differentiate assessment items. Output only a JSON array.",
        op="differentiate", max_tokens=4096,
    )
    if len(parsed) != len(the_items):
        raise ValueError(f"differentiate: expected {len(the_items)} results, got {len(parsed)}")
    out: list[Item] = []
    for it, res in zip(the_items, parsed, strict=True):
        if not isinstance(res, dict) or not str(res.get("stem", "")).strip():
            raise ValueError(f"differentiate: each result needs a 'stem', got {res!r}")
        difficulty = str(res.get("difficulty", it.difficulty))
        if difficulty not in _DIFFICULTIES:
            difficulty = it.difficulty
        out.append(it.model_copy(update={
            "stem": str(res["stem"]).strip(),
            "choices": [str(c) for c in (res.get("choices") or [])] if it.kind == "mcq" else [],
            "answer": str(res.get("answer", it.answer)).strip(),
            "difficulty": difficulty,
            "labels": {**it.labels, "level": level},
        }))
    return ItemSet(items=out)


@operator(library=_LIB, needs_llm=True)
async def translate(passage: Passage, language: str, llm: LLMCaller | None = None) -> Passage:
    """Translate a ``passage`` into ``language`` (e.g. ``"Spanish"``) while preserving meaning and
    reading level — one LLM call. Returns a new :class:`Passage`. Cached per (passage + language).
    """
    assert llm is not None  # guaranteed by the executor for needs_llm ops
    if not language.strip():
        raise ValueError("translate: language must be non-empty")
    prompt = (
        f"Translate this passage into {language}. Preserve meaning, tone, and approximate reading "
        f"level. Return only the translation, no preamble.\n\nTitle: {passage.title}\n\n{passage.text}"
    )
    text = (await llm(prompt, system="You are a bilingual curriculum translator.", max_tokens=2048)).strip()
    return Passage(title=passage.title, text=text, reading_grade=passage.reading_grade,
                   word_count=len(text.split()))


@operator(library=_LIB, needs_llm=True)
async def add_scaffolds(
    items: ItemSet, kind: str = "sentence_starters", llm: LLMCaller | None = None
) -> ItemSet:
    """Add learning **scaffolds** to each item without giving away the answer — one batched LLM call.
    ``kind`` steers the support (e.g. ``sentence_starters``, ``vocabulary``, ``steps``). Appends to
    each item's ``scaffolds``. Errors over ``max_items``. Cached per (items + kind).
    """
    assert llm is not None  # guaranteed by the executor for needs_llm ops
    the_items = items.items
    if not the_items:
        return items
    if len(the_items) > _MAX_ITEMS:
        raise ValueError(f"add_scaffolds: {len(the_items)} items exceeds max_items={_MAX_ITEMS}")
    prompt = (
        f"For each item below, add 1-2 '{kind}' scaffolds that support a struggling student WITHOUT "
        f"revealing the answer. Return ONLY a JSON array of {len(the_items)} objects "
        '{"scaffolds": [<short supports>]}, in order, no prose.\n\n'
        f"Items:\n{_numbered_items(the_items)}"
    )
    parsed = await _llm_json(
        llm, prompt, system="You add instructional scaffolds. Output only a JSON array.",
        op="add_scaffolds", max_tokens=3072,
    )
    if len(parsed) != len(the_items):
        raise ValueError(f"add_scaffolds: expected {len(the_items)} results, got {len(parsed)}")
    out: list[Item] = []
    for it, res in zip(the_items, parsed, strict=True):
        if not isinstance(res, dict):
            raise ValueError(f"add_scaffolds: each result must be an object, got {res!r}")
        extra = [str(s).strip() for s in (res.get("scaffolds") or []) if str(s).strip()]
        out.append(it.model_copy(update={"scaffolds": [*it.scaffolds, *extra]}))
    return ItemSet(items=out)


# --- assess ----------------------------------------------------------------------------------


@operator(library=_LIB, needs_llm=True)
async def build_rubric(objectives: ObjectiveSet, levels: int = 4, llm: LLMCaller | None = None) -> Rubric:
    """Build a scoring rubric for the ``objectives`` with ``levels`` performance levels per criterion
    — one batched LLM call. Cached per (objectives + levels).
    """
    assert llm is not None  # guaranteed by the executor for needs_llm ops
    if not objectives.items:
        raise ValueError("build_rubric: objectives must be non-empty")
    if levels < _MIN_RUBRIC_LEVELS or levels > _MAX_RUBRIC_LEVELS:
        raise ValueError(
            f"build_rubric: levels must be between {_MIN_RUBRIC_LEVELS} and {_MAX_RUBRIC_LEVELS}"
        )
    prompt = (
        f"Create a scoring rubric for these objectives with exactly {levels} performance levels per "
        "criterion (ordered highest → lowest). Return ONLY a JSON array of objects "
        '{"name": <criterion>, "levels": [<one descriptor per level>]}, no prose.\n\n'
        f"Objectives:\n{_objectives_block(objectives.items)}"
    )
    entries = await _llm_json(
        llm, prompt, system="You design assessment rubrics. Output only a JSON array.",
        op="build_rubric", max_tokens=3072,
    )
    criteria: list[RubricCriterion] = []
    for entry in entries:
        if not isinstance(entry, dict) or not str(entry.get("name", "")).strip():
            raise ValueError(f"build_rubric: each entry needs a 'name', got {entry!r}")
        criteria.append(RubricCriterion(
            name=str(entry["name"]).strip(),
            levels=[str(v).strip() for v in (entry.get("levels") or [])],
        ))
    return Rubric(title=f"Rubric ({objectives.standard_code})", criteria=criteria)


@operator(library=_LIB)
async def make_answer_key(items: ItemSet, title: str = "Answer key") -> Report:
    """Render the **teacher-facing** answer key (stem → answer) as a markdown table — a **pure** sink.
    This is the answer artifact for teachers; a bounded tutor has no equivalent ``answer`` move.
    """
    lines = [f"# {title}", ""]
    if not items.items:
        lines.append("_(no items)_")
        return Report(markdown="\n".join(lines))
    lines.append("| # | Item | Answer |")
    lines.append("| --- | --- | --- |")
    for i, it in enumerate(items.items, 1):
        lines.append(f"| {i} | {_cell(it.stem[:120])} | {_cell(it.answer)} |")
    return Report(markdown="\n".join(lines))


@operator(library=_LIB, needs_llm=True)
async def map_to_objectives(items: ItemSet, objectives: ObjectiveSet, llm: LLMCaller | None = None) -> ItemSet:
    """Tag each item with the objective it best assesses (provenance / coverage) — one batched LLM
    call. Writes each item's ``objective``. Errors over ``max_items``. Cached per (items + objectives).
    """
    assert llm is not None  # guaranteed by the executor for needs_llm ops
    the_items = items.items
    if not the_items:
        return items
    if not objectives.items:
        raise ValueError("map_to_objectives: objectives must be non-empty")
    if len(the_items) > _MAX_ITEMS:
        raise ValueError(f"map_to_objectives: {len(the_items)} items exceeds max_items={_MAX_ITEMS}")
    obj_texts = [o.text for o in objectives.items]
    prompt = (
        "For each item below, pick the single objective it best assesses, by index. Return ONLY a JSON "
        f"array of {len(the_items)} objects {{\"objective\": <int index into the objectives>}}, in "
        "order, no prose.\n\n"
        f"Objectives:\n{_objectives_block(objectives.items)}\n\nItems:\n{_numbered_items(the_items)}"
    )
    parsed = await _llm_json(
        llm, prompt, system="You align assessment items to objectives. Output only a JSON array.",
        op="map_to_objectives", max_tokens=2048,
    )
    if len(parsed) != len(the_items):
        raise ValueError(f"map_to_objectives: expected {len(the_items)} results, got {len(parsed)}")
    out: list[Item] = []
    for it, res in zip(the_items, parsed, strict=True):
        idx = res.get("objective") if isinstance(res, dict) else None
        if not isinstance(idx, int) or not (0 <= idx < len(obj_texts)):
            raise ValueError(f"map_to_objectives: objective index out of range: {idx!r}")
        out.append(it.model_copy(update={"objective": obj_texts[idx]}))
    return ItemSet(items=out)


@operator(library=_LIB)
async def check_coverage(items: ItemSet, objectives: ObjectiveSet) -> Report:
    """Report which ``objectives`` have **no** assessment item — a **pure** coverage check (no LLM),
    the teaching analogue of ``find_missing``. Match is by exact objective text (run ``map_to_objectives``
    first so items carry their objective).
    """
    covered = {it.objective for it in items.items if it.objective}
    missing = [o.text for o in objectives.items if o.text not in covered]
    body = (
        "\n".join(f"- {m}" for m in missing)
        if missing
        else "_None — every objective has at least one item._"
    )
    return Report(markdown=f"# Objective coverage\n\n{body}")


# --- bound (emit the tutor contract) ---------------------------------------------------------


@operator(library=_LIB)
async def set_reveal_policy(
    items: ItemSet,
    policy: Literal["never", "on_request", "after_attempts"] = "after_attempts",
    reveal_after_attempts: int = 3,
) -> TutorContract:
    """Open a :class:`TutorContract` seeded from the ``items`` and set its **reveal policy** — a
    **pure** op that starts the *bound* chain. ``policy`` decides whether a tutor may ever surface a
    full solution: ``never``, ``on_request``, or ``after_attempts`` (after ``reveal_after_attempts``
    tries). The contract's objectives and item scope are taken from the items.
    """
    if reveal_after_attempts < 0:
        raise ValueError("set_reveal_policy: reveal_after_attempts must be >= 0")
    objectives = list(dict.fromkeys(it.objective for it in items.items if it.objective))
    item_scope = [it.stem for it in items.items if it.stem][:_MAX_ITEMS]
    return TutorContract(
        objectives=objectives,
        item_scope=item_scope,
        reveal_policy=policy,
        reveal_after_attempts=reveal_after_attempts,
    )


@operator(library=_LIB)
async def define_hint_ladder(contract: TutorContract, rungs: list[str]) -> TutorContract:
    """Set the tutor's **hint ladder** — an ordered list of hints, least → most revealing — on the
    contract. A **pure** op. The tutor may only advance one rung at a time (enforced at tutor time).
    """
    ladder = [HintRung(order=i, text=str(r).strip()) for i, r in enumerate(_as_list(rungs)) if str(r).strip()]
    if not ladder:
        raise ValueError("define_hint_ladder: rungs must be non-empty")
    return contract.model_copy(update={"hint_ladder": ladder})


@operator(library=_LIB)
async def set_escalation_rules(contract: TutorContract, triggers: list[str]) -> TutorContract:
    """Set the conditions under which the tutor must **escalate to the teacher** instead of continuing
    (e.g. ``["3 failed attempts", "signs of distress", "off-topic"]``) — a **pure** op.
    """
    trigs = [str(t).strip() for t in _as_list(triggers) if str(t).strip()]
    if not trigs:
        raise ValueError("set_escalation_rules: triggers must be non-empty")
    return contract.model_copy(update={"escalation_triggers": trigs})


@operator(library=_LIB)
async def set_done_criteria(contract: TutorContract, criteria: str) -> TutorContract:
    """Set what counts as **done** — the observable evidence the student has met the objective (e.g.
    ``"explains the answer in their own words"``) — a **pure** op that completes the contract.
    """
    if not criteria.strip():
        raise ValueError("set_done_criteria: criteria must be non-empty")
    return contract.model_copy(update={"done_criteria": criteria.strip()})


@operator(library=_LIB)
async def export_contract(contract: TutorContract) -> Report:
    """Render a :class:`TutorContract` as a human-readable markdown brief for the teacher's approval
    gate — a **pure** sink. The contract artifact itself (not this report) is the boundary a student
    tutor would load.
    """
    lines = [
        "# Tutor contract", "",
        "## Objectives",
        *([f"- {o}" for o in contract.objectives] or ["_none_"]), "",
        "## Reveal policy",
        f"- Policy: **{contract.reveal_policy}**"
        + (f" (after {contract.reveal_after_attempts} attempts)" if contract.reveal_policy == "after_attempts" else ""),
        "",
        "## Hint ladder (least → most revealing)",
        *([f"{r.order + 1}. {r.text}" for r in contract.hint_ladder] or ["_none defined_"]), "",
        "## Escalation triggers",
        *([f"- {t}" for t in contract.escalation_triggers] or ["_none defined_"]), "",
        "## Done criteria",
        f"{contract.done_criteria or '_none defined_'}", "",
        "## Moves",
        f"- Allowed: {', '.join(contract.allowed_moves)}",
        f"- Forbidden: {', '.join(contract.forbidden_moves)}",
        "",
        f"_Item scope: {len(contract.item_scope)} item(s)._",
    ]
    return Report(markdown="\n".join(lines))


# --- assemble --------------------------------------------------------------------------------


@operator(library=_LIB)
async def assemble_unit(first: Report, second: Report, title: str = "Lesson unit") -> Report:
    """Concatenate two reports into one packet under a title — a **pure** sink for a multi-part
    deliverable (passage + worksheet + key). Pipe the first in and wire the second as a kwarg; chain
    to combine three or more.
    """
    return Report(
        markdown=f"# {title}\n\n{first.markdown.strip()}\n\n---\n\n{second.markdown.strip()}"
    )


# --- domain preset blocks (thin defaults over the neutral ops) -----------------------------
# NB: no ``reads_file`` operator appears inside a block — content-fingerprint autofill runs on the
# authored (pre-expansion) graph, so a file-reading op must be authored directly. (This métier has
# no file loaders in v1, so the constraint is moot but preserved for when one is added.)


def _reading_lesson() -> Graph:
    g = Graph()
    g.add(Node(id="p", op="generate_passage", params={"reading_grade": 5, "words": 400}))
    g.add(Node(id="i", op="generate_items", params={"count": 6, "kinds": ["mcq", "short"]}, inputs={"source": "p"}))
    g.add(Node(id="w", op="build_worksheet", params={"title": "Reading worksheet"}, inputs={"items": "i"}))
    return g


def _quiz_from_standard() -> Graph:
    g = Graph()
    g.add(Node(id="o", op="set_objectives", params={"count": 4}))
    g.add(Node(id="i", op="generate_items", params={"count": 8, "kinds": ["mcq"]}, inputs={"source": "o"}))
    g.add(Node(id="q", op="build_quiz", params={"title": "Quiz"}, inputs={"items": "i"}))
    return g


def _differentiated_worksheet() -> Graph:
    g = Graph()
    g.add(Node(id="i", op="generate_items", params={"count": 6, "kinds": ["short"]}))
    g.add(Node(id="d", op="differentiate", params={"level": "support"}, inputs={"items": "i"}))
    g.add(Node(id="w", op="build_worksheet", params={"title": "Differentiated worksheet"}, inputs={"items": "d"}))
    return g


def _tutor_contract() -> Graph:
    g = Graph()
    g.add(Node(id="rp", op="set_reveal_policy",
               params={"policy": "after_attempts", "reveal_after_attempts": 3}))
    g.add(Node(id="hl", op="define_hint_ladder",
               params={"rungs": ["Restate the question in your own words",
                                 "Point to the part of the material that helps",
                                 "Give the first step, not the answer"]},
               inputs={"contract": "rp"}))
    # NB: the escalation attempt-limit (5) is set ABOVE reveal_after_attempts (3) so the two thresholds
    # don't collide — otherwise escalation would always pre-empt the `after_attempts` reveal.
    g.add(Node(id="er", op="set_escalation_rules",
               params={"triggers": ["5 failed attempts", "signs of distress", "off-topic"]},
               inputs={"contract": "hl"}))
    g.add(Node(id="dc", op="set_done_criteria",
               params={"criteria": "explains the answer in their own words"},
               inputs={"contract": "er"}))
    return g


_BLOCKS = [
    BlockDef(
        name="reading_lesson", graph=_reading_lesson(), library=_LIB,
        description="generate a passage, questions, and a worksheet from objectives (reading comprehension)",
        input_ports={"objectives": ("p", "objectives")},
        params={"reading_grade": ("p", "reading_grade"), "words": ("p", "words"),
                "count": ("i", "count"), "kinds": ("i", "kinds"), "title": ("w", "title")},
        output="w",
    ),
    BlockDef(
        name="quiz_from_standard", graph=_quiz_from_standard(), library=_LIB,
        description="draft objectives from a standard, then a quiz (standalone questions)",
        input_ports={"standard": ("o", "standard")},
        params={"count": ("o", "count"), "focus": ("o", "focus"),
                "item_count": ("i", "count"), "kinds": ("i", "kinds"), "title": ("q", "title")},
        output="q",
    ),
    BlockDef(
        name="differentiated_worksheet", graph=_differentiated_worksheet(), library=_LIB,
        description="generate items, differentiate them for a level, and build a worksheet",
        input_ports={"source": ("i", "source")},
        params={"count": ("i", "count"), "kinds": ("i", "kinds"),
                "level": ("d", "level"), "title": ("w", "title")},
        output="w",
    ),
    BlockDef(
        name="tutor_contract", graph=_tutor_contract(), library=_LIB,
        description="one-click TutorContract from items: reveal policy → hint ladder → escalation → done",
        input_ports={"items": ("rp", "items")},
        params={"policy": ("rp", "policy"), "reveal_after_attempts": ("rp", "reveal_after_attempts"),
                "rungs": ("hl", "rungs"), "triggers": ("er", "triggers"), "criteria": ("dc", "criteria")},
        output="dc",
    ),
]


def register() -> None:
    """Register the teaching preset blocks (idempotent). Operators register on import above."""
    for block in _BLOCKS:
        if not is_block(block.name):
            register_block(block)


register()
