"""Single entry point for all aether gates — one subprocess per bash call."""

from __future__ import annotations

import concurrent.futures
import json
import sys
from collections.abc import Callable
from typing import Any

from .bonsai import evaluate as bonsai
from .cairn import evaluate as cairn
from .install import maybe_refresh_skills
from .temper import evaluate as temper
from .whetstone import evaluate as whetstone

_GATES = (whetstone, bonsai, temper, cairn)
_GATE_TIMEOUT_S = 2.0


def _run_gate(gate: Callable[..., Any], command: str, cwd: str) -> dict | None:
    """Run a gate with a timeout; returns None on timeout or error."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(gate, command, cwd)
        try:
            return future.result(timeout=_GATE_TIMEOUT_S)
        except concurrent.futures.TimeoutError:
            return None
        except Exception:
            return None


def main() -> None:
    maybe_refresh_skills()

    try:
        invocation = json.loads(sys.stdin.buffer.read())
        command = invocation.get("tool_input", {}).get("command", "")
        cwd = invocation.get("cwd", ".")
    except Exception:
        sys.exit(0)

    for gate in _GATES:
        result = _run_gate(gate, command, cwd)
        if result:
            print(json.dumps(result))
            return


if __name__ == "__main__":
    main()
