from __future__ import annotations

import pytest

from vibe.core.graph.executor import GraphValidationError, execute
import vibe.core.graph.library.teaching as T
from vibe.core.graph.model import Graph, Node


def _objectives() -> T.ObjectiveSet:
    return T.ObjectiveSet(
        standard_code="STD.1",
        items=[
            T.Objective(text="Identify the main idea of a text", standard_code="STD.1"),
            T.Objective(text="Summarize a text in one sentence", standard_code="STD.1"),
        ],
    )


def _items() -> T.ItemSet:
    return T.ItemSet(items=[
        T.Item(stem="What is the main idea?", kind="short", answer="The ecosystem cycles matter.",
               objective="Identify the main idea of a text"),
        T.Item(stem="Which choice best summarizes?", kind="mcq", choices=["A", "B", "C", "D"],
               answer="B", objective="Summarize a text in one sentence"),
    ])


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
async def test_set_standard_and_sample_standard() -> None:
    s = await T.set_standard(code="X.1", subject="Math", grade="6")
    assert s.code == "X.1" and s.subject == "Math"
    with pytest.raises(ValueError, match="code must be non-empty"):
        await T.set_standard(code="  ")
    sample = await T.sample_standard(name="ccss_ela_5")
    assert sample.code.startswith("CCSS") and sample.grade == "5"
    with pytest.raises(ValueError, match="unknown sample standard"):
        await T.sample_standard(name="nope")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_build_worksheet_hides_answers_and_shows_choices() -> None:
    rep = await T.build_worksheet(_items(), title="WS")
    assert rep.markdown.startswith("# WS")
    assert "What is the main idea?" in rep.markdown
    assert "A. A" in rep.markdown  # mcq choices are lettered
    assert "The ecosystem cycles matter." not in rep.markdown  # answer is hidden


@pytest.mark.asyncio
async def test_build_quiz_tags_kind_and_difficulty() -> None:
    rep = await T.build_quiz(_items(), title="Q")
    assert "_(short, medium)_" in rep.markdown and "B" not in rep.markdown.split("choice")[0][:5]


@pytest.mark.asyncio
async def test_make_answer_key_is_teacher_facing() -> None:
    rep = await T.make_answer_key(_items())
    assert "| Answer |" in rep.markdown and "The ecosystem cycles matter." in rep.markdown


@pytest.mark.asyncio
async def test_check_coverage_flags_uncovered_objectives() -> None:
    items = T.ItemSet(items=[T.Item(stem="q", objective="Identify the main idea of a text")])
    rep = await T.check_coverage(items, _objectives())
    assert "Summarize a text in one sentence" in rep.markdown  # uncovered
    assert "Identify the main idea of a text" not in rep.markdown  # covered


@pytest.mark.asyncio
async def test_assemble_unit_merges() -> None:
    out = await T.assemble_unit(T.Report(markdown="# A\naaa"), T.Report(markdown="# B\nbbb"), title="Unit")
    assert "# Unit" in out.markdown and "aaa" in out.markdown and "bbb" in out.markdown


# --- bound ops (pure) — the contract chain --------------------------------------------------


@pytest.mark.asyncio
async def test_contract_chain_builds_a_tutor_contract() -> None:
    c = await T.set_reveal_policy(_items(), policy="after_attempts", reveal_after_attempts=2)
    assert isinstance(c, T.TutorContract)
    assert c.reveal_policy == "after_attempts" and c.reveal_after_attempts == 2
    assert set(c.objectives) == {"Identify the main idea of a text", "Summarize a text in one sentence"}
    assert len(c.item_scope) == 2
    assert "answer" in c.forbidden_moves and "answer" not in c.allowed_moves

    c = await T.define_hint_ladder(c, rungs=["restate it", "point to the line", "give the first step"])
    assert [r.order for r in c.hint_ladder] == [0, 1, 2]
    c = await T.set_escalation_rules(c, triggers=["3 failed attempts", "distress"])
    assert "distress" in c.escalation_triggers
    c = await T.set_done_criteria(c, criteria="explains it in their own words")
    assert c.done_criteria == "explains it in their own words"

    rep = await T.export_contract(c)
    assert "# Tutor contract" in rep.markdown and "give the first step" in rep.markdown
    assert "Forbidden: answer" in rep.markdown


@pytest.mark.asyncio
async def test_bound_ops_reject_empty_input() -> None:
    c = await T.set_reveal_policy(_items())
    with pytest.raises(ValueError, match="rungs must be non-empty"):
        await T.define_hint_ladder(c, rungs=[])
    with pytest.raises(ValueError, match="triggers must be non-empty"):
        await T.set_escalation_rules(c, triggers=[])
    with pytest.raises(ValueError, match="criteria must be non-empty"):
        await T.set_done_criteria(c, criteria="  ")


# --- LLM-backed ops ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_objectives_builds_objectives() -> None:
    llm = _StubLLM('[{"text": "Identify main idea", "bloom_level": "understand"}, '
                   '{"text": "Summarize", "bloom_level": "understand"}]')
    out = await T.set_objectives(await T.sample_standard(name="ccss_ela_5"), count=2, llm=llm)
    assert [o.text for o in out.items] == ["Identify main idea", "Summarize"]
    assert all(o.standard_code.startswith("CCSS") for o in out.items)
    with pytest.raises(ValueError, match="count must be between"):
        await T.set_objectives(await T.sample_standard(name="ccss_ela_5"), count=0, llm=llm)


@pytest.mark.asyncio
async def test_generate_items_from_objectives_and_passage() -> None:
    reply = ('[{"stem": "Q1?", "kind": "mcq", "choices": ["a","b","c","d"], "answer": "a", '
             '"objective": "main idea", "difficulty": "easy"}]')
    out = await T.generate_items(_objectives(), count=1, kinds=["mcq"], llm=_StubLLM(reply))
    assert out.items[0].kind == "mcq" and out.items[0].choices == ["a", "b", "c", "d"]
    # a Passage source is accepted too (union input)
    psg = T.Passage(title="Eco", text="Energy flows through an ecosystem.", reading_grade=5, word_count=6)
    out2 = await T.generate_items(psg, count=1, kinds=["mcq"], llm=_StubLLM(reply))
    assert out2.items[0].stem == "Q1?"


@pytest.mark.asyncio
async def test_generate_items_rejects_bad_kind() -> None:
    with pytest.raises(ValueError, match="unknown item kind"):
        await T.generate_items(_objectives(), kinds=["truefalse"], llm=_StubLLM("[]"))
    # a returned kind outside the requested set is rejected
    with pytest.raises(ValueError, match="not in"):
        await T.generate_items(
            _objectives(), count=1, kinds=["short"],
            llm=_StubLLM('[{"stem": "Q?", "kind": "essay", "answer": "x"}]'),
        )


@pytest.mark.asyncio
async def test_generate_passage_measures_word_count() -> None:
    llm = _StubLLM("The Water Cycle\nWater moves from sea to sky to land again.")
    p = await T.generate_passage(_objectives(), reading_grade=4, words=50, llm=llm)
    assert p.title == "The Water Cycle" and p.reading_grade == 4
    assert p.word_count == len("Water moves from sea to sky to land again.".split())


@pytest.mark.asyncio
async def test_differentiate_preserves_kind_and_labels_level() -> None:
    llm = _StubLLM('[{"stem": "Easier main idea?", "answer": "eco", "difficulty": "easy"}, '
                   '{"stem": "Easier summary?", "choices": ["A","B","C","D"], "answer": "B", "difficulty": "easy"}]')
    out = await T.differentiate(_items(), level="support", llm=llm)
    assert out.items[0].stem == "Easier main idea?" and out.items[0].difficulty == "easy"
    assert out.items[1].kind == "mcq" and out.items[1].choices == ["A", "B", "C", "D"]
    assert all(it.labels["level"] == "support" for it in out.items)


@pytest.mark.asyncio
async def test_add_scaffolds_appends() -> None:
    llm = _StubLLM('[{"scaffolds": ["Start with: The main idea is…"]}, {"scaffolds": ["Reread paragraph 1"]}]')
    out = await T.add_scaffolds(_items(), kind="sentence_starters", llm=llm)
    assert out.items[0].scaffolds == ["Start with: The main idea is…"]


@pytest.mark.asyncio
async def test_build_rubric_builds_criteria() -> None:
    llm = _StubLLM('[{"name": "Main idea", "levels": ["Clear", "Partial", "Unclear"]}]')
    r = await T.build_rubric(_objectives(), levels=3, llm=llm)
    assert r.criteria[0].name == "Main idea" and len(r.criteria[0].levels) == 3
    with pytest.raises(ValueError, match="levels must be between"):
        await T.build_rubric(_objectives(), levels=1, llm=llm)


@pytest.mark.asyncio
async def test_map_to_objectives_tags_items() -> None:
    llm = _StubLLM('[{"objective": 0}, {"objective": 1}]')
    plain = T.ItemSet(items=[T.Item(stem="q1"), T.Item(stem="q2")])
    out = await T.map_to_objectives(plain, _objectives(), llm=llm)
    assert out.items[0].objective == "Identify the main idea of a text"
    assert out.items[1].objective == "Summarize a text in one sentence"
    with pytest.raises(ValueError, match="index out of range"):
        await T.map_to_objectives(plain, _objectives(), llm=_StubLLM('[{"objective": 9}, {"objective": 0}]'))


@pytest.mark.asyncio
async def test_sequence_lesson_returns_prose() -> None:
    out = await T.sequence_lesson(_objectives(), minutes=30, llm=_StubLLM("## Overview\nA plan."))
    assert out.markdown.startswith("## Overview")


@pytest.mark.asyncio
async def test_llm_json_retries_then_errors_on_empty() -> None:
    std = await T.sample_standard(name="ccss_ela_5")
    # empty first, valid on the retry → succeeds
    seq = _SeqStubLLM("", '[{"text": "Identify main idea", "bloom_level": "understand"}]')
    out = await T.set_objectives(std, count=1, llm=seq)
    assert len(out.items) == 1 and seq.calls == 2
    # always empty → actionable error
    with pytest.raises(ValueError, match="returned no content"):
        await T.set_objectives(std, count=1, llm=_StubLLM(""))


# --- executor / registry contract ----------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_op_needs_a_backend() -> None:
    g = Graph()
    g.add(Node(id="s", op="sample_standard", params={"name": "ccss_ela_5"}))
    g.add(Node(id="o", op="set_objectives", params={"count": 2}, inputs={"standard": "s"}))
    with pytest.raises(GraphValidationError, match="needs an LLM"):
        await execute(g, None, llm=None)


def test_llm_ops_hide_the_reserved_param() -> None:
    from vibe.core.graph.operators import get_operator

    for name in ("set_objectives", "sequence_lesson", "generate_passage", "generate_items",
                 "adapt_reading_level", "differentiate", "translate", "add_scaffolds",
                 "build_rubric", "map_to_objectives"):
        spec = get_operator(name)
        assert spec.needs_llm and "llm" not in spec.param_names and "llm" not in spec.arg_types


def test_pure_ops_are_not_llm() -> None:
    from vibe.core.graph.operators import get_operator

    for name in ("set_standard", "sample_standard", "build_worksheet", "build_quiz",
                 "make_answer_key", "check_coverage", "assemble_unit", "set_reveal_policy",
                 "define_hint_ladder", "set_escalation_rules", "set_done_criteria", "export_contract"):
        assert not get_operator(name).needs_llm


def test_all_ops_tagged_teaching_library() -> None:
    from vibe.core.graph.operators import get_operator

    for name in ("set_standard", "generate_items", "set_reveal_policy", "export_contract"):
        assert get_operator(name).library == "teaching"


def test_blocks_are_registered() -> None:
    from vibe.core.graph.blocks import is_block

    assert all(is_block(n) for n in
               ("reading_lesson", "quiz_from_standard", "differentiated_worksheet", "tutor_contract"))
