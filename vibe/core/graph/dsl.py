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

# DOTALL so `.*` spans newlines — a `name = …` whose body holds a multi-line sql(query=\"\"\"…\"\"\")
# still matches (without it, `$` fails mid-string and the assignment is mis-parsed as a call).
_ASSIGN_RE = re.compile(r"^([A-Za-z_]\w*)\s*=\s*(.*)$", re.DOTALL)
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


def _scan_lines(text: str) -> list[str]:
    """Split ``text`` into logical lines at newlines outside any string AND at bracket depth 0.

    A newline inside a quoted string (a multi-line ``sql(query=\"\"\"…\"\"\")``) or inside an open
    ``(``/``[``/``{`` (a call whose args span lines) does not end a line — a multi-line ``op(...)``
    stays one statement. A ``#`` at depth 0 starts a comment. Raises on an unterminated string.
    """
    lines: list[str] = []
    quote: str | None = None
    depth = 0  # open ( [ { — a newline inside a call/list keeps the statement going
    buf: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if quote:
            if ch == "\\" and len(quote) == 1 and i + 1 < n:  # escape (single-char quotes only)
                buf.append(ch + text[i + 1])
                i += 2
            elif text.startswith(quote, i):
                buf.append(quote)
                i += len(quote)
                quote = None
            else:
                buf.append(ch)  # any char (incl. newline) inside the string is literal
                i += 1
            continue
        if ch in "\"'":
            quote = text[i : i + 3] if text[i : i + 3] in {'"""', "'''"} else ch
            buf.append(quote)
            i += len(quote)
        elif ch in "([{":
            depth += 1
            buf.append(ch)
            i += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
            buf.append(ch)
            i += 1
        elif ch == "#" and depth == 0:  # line comment (outside a string) — skip to the newline
            while i < n and text[i] != "\n":
                i += 1
        elif ch == "\n":
            if depth == 0:
                lines.append("".join(buf))
                buf = []
            else:
                buf.append(" ")  # newline inside a call → whitespace; statement continues
            i += 1
        else:
            buf.append(ch)
            i += 1
    if quote:
        raise DSLError(f"unterminated string in {text!r}")
    lines.append("".join(buf))
    return lines


def _statements(text: str) -> list[str]:
    """Split a program into statements, quote- and continuation-aware.

    Lines are split outside strings (see :func:`_scan_lines`), then folded: a line starting with
    ``|`` — or a line following one that ends with a dangling ``|`` — continues the previous
    statement, so the readable multi-line pipeline form parses like a single line. A line that is
    itself an assignment (``name = …``) always starts its own statement.
    """
    statements: list[str] = []
    for line in _scan_lines(text):
        stripped = line.strip()
        if not stripped:
            continue
        # Fold a continuation onto the previous statement when the pipe sits at either boundary:
        # this line starts with `|` (always a continuation), or the previous statement ended with a
        # dangling `|` — EXCEPT never fold a line that is itself an assignment (`name = …`), which
        # must start its own statement (otherwise `foo |` + `bar = baz` glues into one broken line).
        starts_pipe = stripped.startswith("|")
        prev_dangling = bool(statements) and statements[-1].endswith("|")
        is_assignment = _ASSIGN_RE.match(stripped) is not None
        if statements and (starts_pipe or (prev_dangling and not is_assignment)):
            statements[-1] = f"{statements[-1]} {stripped}"
        else:
            statements.append(stripped)
    return statements


# Tools the analyst calls on their own — not pipeline operators. Named here only to give a clear
# hint when one is mistakenly used as a step (the DSL itself knows nothing about tools).
_STANDALONE_TOOLS = frozenset({"graph_inspect", "graph_save_block", "ask_user_question"})


def _input_ports(op: str) -> tuple[str, ...]:
    if is_block(op):
        return tuple(get_block(op).input_ports)
    if is_registered(op):
        return get_operator(op).input_names
    if op in _STANDALONE_TOOLS:
        raise DSLError(f"{op!r} is a separate tool — call it on its own, not as a pipeline step")
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
            if not arg.strip():  # tolerate a trailing (or doubled) comma
                continue
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
    # Tolerate a single dangling trailing `|` (agent started a new statement after it); an empty
    # interior/leading segment is a real mistake (`||` or a stray leading `|`).
    while len(segments) > 1 and segments[-1].strip() == "":
        segments.pop()
    prev: str | None = None
    for idx, raw_seg in enumerate(segments):
        seg = raw_seg.strip()
        if not seg:
            raise DSLError("empty step in pipeline (check for '||' or a stray '|')")
        if _IDENT_RE.match(seg):  # a bare reference to an earlier node (pipeline start only)
            if idx != 0:
                raise DSLError(
                    f"reference {seg!r} may only start a pipeline — to reuse a step, name it "
                    f"(`x = …`) then start a new line with `x | …`, or wire it as an input kwarg "
                    f"(e.g. `sql(query=…, t2={seg})`)"
                )
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
