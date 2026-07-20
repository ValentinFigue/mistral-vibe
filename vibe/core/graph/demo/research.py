"""A research-shaped operator kit for the comparison demo.

Same graph *shape* as an LLM research agent — fetch → extract → summarize → synthesize —
but every operator is a **deterministic stand-in** (no LLM, no network, no API keys), so the
comparison is repeatable and cache-sound. ``summarize_passages`` is extractive (first-N
sentences), a placeholder for a future LLM ``generate`` operator (see the backlog). The point
here is to show the context / recompute wins hold on a research-shaped pipeline, not to do
real NLP.

The DAG (:func:`build_research_graph`)::

    fetch_docs(seed) ─ extract_passages(keyword) ─ summarize_passages(max_sentences) ─ synthesize_brief(title)
"""

from __future__ import annotations

import random

from pydantic import BaseModel, Field

from vibe.core.graph.model import Graph, Node
from vibe.core.graph.operators import operator

_TOPICS = ("latency", "pricing", "onboarding", "reliability", "security", "roadmap")
_VERBS = ("improved", "regressed", "stabilized", "spiked", "dropped", "held steady")
_SUBJECTS = ("the checkout flow", "the mobile app", "the API", "the dashboard", "billing")


class DocSet(BaseModel):
    """A synthetic corpus — a list of documents, each a block of sentences."""

    docs: list[str] = Field(default_factory=list)


class Passages(BaseModel):
    sentences: list[str] = Field(default_factory=list)


class Summary(BaseModel):
    sentences: list[str] = Field(default_factory=list)


class Brief(BaseModel):
    markdown: str


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


@operator
async def fetch_docs(seed: int, n_docs: int, sentences_per_doc: int) -> DocSet:
    """Generate a synthetic corpus (deterministic in ``seed``). Stands in for web fetches."""
    rng = _rng(seed)
    docs: list[str] = []
    for d in range(n_docs):
        sentences = [
            f"In doc {d}, {rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} on {rng.choice(_TOPICS)}."
            for _ in range(sentences_per_doc)
        ]
        docs.append(" ".join(sentences))
    return DocSet(docs=docs)


@operator
async def extract_passages(docs: DocSet, keyword: str) -> Passages:
    """Extract every sentence mentioning ``keyword`` across the corpus."""
    sentences: list[str] = []
    for doc in docs.docs:
        for sentence in doc.split(". "):
            if keyword.lower() in sentence.lower():
                sentences.append(sentence.strip().rstrip(".") + ".")
    return Passages(sentences=sentences)


@operator
async def summarize_passages(passages: Passages, max_sentences: int) -> Summary:
    """Extractive summary: the first ``max_sentences`` passages.

    A deterministic placeholder for an LLM ``generate`` operator — see the module docstring.
    """
    return Summary(sentences=passages.sentences[:max_sentences])


@operator
async def synthesize_brief(summary: Summary, title: str) -> Brief:
    """Render the summary sentences as a markdown brief."""
    lines = [f"# {title}", ""]
    lines.extend(f"- {sentence}" for sentence in summary.sentences)
    return Brief(markdown="\n".join(lines))


def build_research_graph(
    *,
    seed: int = 11,
    n_docs: int = 400,
    sentences_per_doc: int = 8,
    keyword: str = "latency",
    max_sentences: int = 8,
    title: str = "Latency research brief",
) -> Graph:
    """Wire the fetch → extract → summarize → synthesize research DAG."""
    graph = Graph()
    graph.add(
        Node(
            id="docs",
            op="fetch_docs",
            params={"seed": seed, "n_docs": n_docs, "sentences_per_doc": sentences_per_doc},
        )
    )
    graph.add(
        Node(id="passages", op="extract_passages", params={"keyword": keyword}, inputs={"docs": "docs"})
    )
    graph.add(
        Node(
            id="summary",
            op="summarize_passages",
            params={"max_sentences": max_sentences},
            inputs={"passages": "passages"},
        )
    )
    graph.add(
        Node(id="brief", op="synthesize_brief", params={"title": title}, inputs={"summary": "summary"})
    )
    return graph
