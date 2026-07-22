"""A compact pipeline DSL that parses to a Métier :class:`Graph`.

The analyst authors an analysis as a short program instead of typed patch JSON. Each line is
either an assignment ``name = pipeline`` or a bare ``pipeline``; a pipeline is steps joined by
``|``. A step is an operator/block call ``op(key=value, …)`` (or a bare name referencing an
earlier step, only as a pipeline's first segment). ``|`` wires the previous step's output into
the next step's **first input port**; a kwarg whose name is an input port and whose value is a
bare reference wires that port (e.g. ``sql(query=…, t2=customers)`` / ``join(right=other, …)``).
Everything else is a literal param. `sql(...)` carries the heavy computation, so the grammar
stays deliberately small.

The result is an ordinary `Graph` — the caching, handles, and incremental recompute are the same
as hand-authored patches.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
import itertools
import re

from vibe.core.graph.blocks import get_block, is_block
from vibe.core.graph.model import Graph, Node
from vibe.core.graph.operators import get_operator, is_registered

_ASSIGN_RE = re.compile(r"^([A-Za-z_]\w*)\s*=\s*(.*)$")
_CALL_RE = re.compile(r"^([A-Za-z_]\w*)\s*\((.*)\)$", re.DOTALL)
_IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")


class DSLError(ValueError):
    """The pipeline program is malformed."""


def _split_top_level(text: str, sep: str) -> list[str]:
    """Split on ``sep`` outside of (), [], and quotes.

    Strings may be single- or double-quoted, or **triple-quoted** (``\"\"\"…\"\"\"`` / ``'''…'''``)
    — the triple form lets an embedded ``sql(query=…)`` carry unescaped single quotes, double
    quotes, commas, and ``|`` without tripping the splitter.
    """
    parts: list[str] = []
    depth = 0
    quote: str | None = None  # active closing delimiter: ' " ''' or \"\"\"
    buf: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if quote:
            if ch == "\\" and len(quote) == 1 and i + 1 < n:  # escape (single-char quotes only)
                buf.append(ch)
                buf.append(text[i + 1])
                i += 2
                continue
            if text.startswith(quote, i):  # closing delimiter (1 or 3 chars)
                buf.append(quote)
                i += len(quote)
                quote = None
                continue
            buf.append(ch)
            i += 1
            continue
        if ch in "\"'":
            quote = text[i : i + 3] if text[i : i + 3] in {'"""', "'''"} else ch
            buf.append(quote)
            i += len(quote)
            continue
        if ch in "([":
            depth += 1
            buf.append(ch)
        elif ch in ")]":
            depth -= 1
            buf.append(ch)
        elif ch == sep and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    if quote:
        raise DSLError(f"unterminated string in {text!r}")
    parts.append("".join(buf))
    return parts


def _statements(text: str) -> list[str]:
    """Split a program into statements, quote- and continuation-aware.

    A newline ends a statement only when it is **outside** any quoted string, so a multi-line
    ``sql(query=\"\"\"…\"\"\")`` stays one statement. A ``#`` outside a string starts a line
    comment. Finally, a statement beginning with ``|`` is folded onto the previous one, so the
    readable multi-line pipeline form (leading-``|`` continuations) parses like a single line.
    """
    lines: list[str] = []
    quote: str | None = None
    buf: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if quote:
            if ch == "\\" and len(quote) == 1 and i + 1 < n:  # escape (single-char quotes only)
                buf.append(ch)
                buf.append(text[i + 1])
                i += 2
                continue
            if text.startswith(quote, i):
                buf.append(quote)
                i += len(quote)
                quote = None
                continue
            buf.append(ch)  # any char (incl. newline) inside the string is literal
            i += 1
            continue
        if ch in "\"'":
            quote = text[i : i + 3] if text[i : i + 3] in {'"""', "'''"} else ch
            buf.append(quote)
            i += len(quote)
            continue
        if ch == "#":  # line comment (outside a string) — skip to the newline, keep the newline
            while i < n and text[i] != "\n":
                i += 1
            continue
        if ch == "\n":
            lines.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    if quote:
        raise DSLError(f"unterminated string in {text!r}")
    lines.append("".join(buf))

    statements: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Fold onto the previous statement when the pipe sits at either boundary: this line
        # starts with `|`, or the previous line ended with a dangling `|`. (A `|` inside a string
        # is safe — a newline inside a quote never ended a statement here.)
        if statements and (stripped.startswith("|") or statements[-1].endswith("|")):
            statements[-1] = f"{statements[-1]} {stripped}"
        else:
            statements.append(stripped)
    return statements


def _input_ports(op: str) -> tuple[str, ...]:
    if is_block(op):
        return tuple(get_block(op).input_ports)
    if is_registered(op):
        return get_operator(op).input_names
    raise DSLError(f"unknown operator or block {op!r}")


def _parse_value(token: str) -> tuple[str, object]:
    """Return ``("ref", name)`` for a bare identifier, else ``("lit", python_value)``."""
    tok = token.strip()
    if _IDENT_RE.match(tok) and tok not in {"true", "false"}:
        return "ref", tok
    try:
        return "lit", ast.literal_eval(tok)  # numbers, "strings", [lists], True/False via below
    except (ValueError, SyntaxError):
        if tok in {"true", "false"}:
            return "lit", tok == "true"
        raise DSLError(f"cannot parse value {token!r}") from None


def _parse_call(segment: str) -> tuple[str, list[tuple[str, str]]]:
    m = _CALL_RE.match(segment.strip())
    if not m:
        raise DSLError(f"expected `op(...)`, got {segment.strip()!r}")
    op, argstr = m.group(1), m.group(2).strip()
    kwargs: list[tuple[str, str]] = []
    if argstr:
        for arg in _split_top_level(argstr, ","):
            key, sep, val = arg.partition("=")
            if not sep or not _IDENT_RE.match(key.strip()):
                raise DSLError(f"argument must be `key=value` in {op!r}, got {arg.strip()!r}")
            kwargs.append((key.strip(), val.strip()))
    return op, kwargs


def _step_io(op: str, kwargs: list[tuple[str, str]], named: dict[str, str]) -> tuple[dict, dict]:
    """Split a call's kwargs into (params, inputs) — a ref into an input port wires a node."""
    ports = _input_ports(op)
    params: dict[str, object] = {}
    inputs: dict[str, str] = {}
    for key, raw in kwargs:
        kind, value = _parse_value(raw)
        if kind == "ref":
            ref = str(value)  # a ref token is always the matched identifier string
            if key not in ports:
                raise DSLError(f"{op!r}: {key!r} is not an input port (ports: {list(ports)})")
            if ref not in named:
                raise DSLError(f"unknown reference {ref!r}")
            inputs[key] = named[ref]
        else:
            params[key] = value
    return params, inputs


def _add_statement(
    graph: Graph, name: str | None, body: str, named: dict[str, str], counter: Iterator[int]
) -> None:
    """Parse one `[name =] pipeline` statement, adding its nodes and recording the name."""
    segments = _split_top_level(body, "|")
    prev: str | None = None
    for idx, raw_seg in enumerate(segments):
        seg = raw_seg.strip()
        if _IDENT_RE.match(seg):  # a bare reference to an earlier node (pipeline start only)
            if idx != 0:
                raise DSLError(f"reference {seg!r} may only start a pipeline")
            if seg not in named:
                raise DSLError(f"unknown reference {seg!r}")
            prev = named[seg]
            continue
        op, kwargs = _parse_call(seg)
        is_last = idx == len(segments) - 1
        node_id: str = name if (name is not None and is_last) else f"s{next(counter)}"
        if node_id in graph.nodes:
            raise DSLError(f"duplicate step name {node_id!r}")
        params, inputs = _step_io(op, kwargs, named)
        if prev is not None:  # wire the piped input into the first input port
            ports = _input_ports(op)
            if not ports:
                raise DSLError(f"{op!r} takes no piped input")
            inputs.setdefault(ports[0], prev)
        graph.add(Node(id=node_id, op=op, params=params, inputs=inputs))
        prev = node_id
    if name:
        if prev is None:
            raise DSLError(f"assignment {name!r} has no step")
        named[name] = prev


def parse_pipeline(text: str) -> Graph:
    """Parse a DSL program into a :class:`Graph`. Raises :class:`DSLError` on any problem.

    A pipeline may be written across lines for readability: a line beginning with ``|`` is a
    continuation folded onto the previous statement, and a multi-line ``sql(query=\"\"\"…\"\"\")``
    stays a single statement (newlines inside a string do not split it). ``#`` starts a line
    comment. See :func:`_statements`.
    """
    graph = Graph()
    named: dict[str, str] = {}  # assigned name -> node id
    counter = itertools.count(1)
    statements = _statements(text)
    if not statements:
        raise DSLError("empty pipeline")
    for stmt in statements:
        assign = _ASSIGN_RE.match(stmt)
        name, body = (assign.group(1), assign.group(2)) if assign else (None, stmt)
        _add_statement(graph, name, body, named, counter)
    return graph
