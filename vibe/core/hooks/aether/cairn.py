"""Gate: nudge toward better commit messages and CHANGELOG hygiene."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .bypass import is_bypassed

_MSG_PATTERN = re.compile(
    r'(?:--message=|--message\s+|-m\s+)(?:"([^"]+)"|\'([^\']+)\'|(\S+))',
)
_WEAK = re.compile(
    r"^(fix|wip|update|change|edit|minor|small|quick|temp|test|stuff|thing"
    r"|work|done|ok|oops|misc|cleanup|changes?|tweak|patch)\s*[.!]*$",
    re.IGNORECASE,
)
# File extensions that count as "non-trivial" source changes (not docs/config).
_SOURCE_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".kt",
    ".swift", ".c", ".cpp", ".h", ".cs", ".rb", ".php",
}


def _staged_files(cwd: str) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True, text=True, cwd=cwd, timeout=2,
        )
        return [f.strip() for f in result.stdout.splitlines() if f.strip()]
    except Exception:
        return []


def _changelog_warning(cwd: str) -> dict | None:
    """Return a warn result if CHANGELOG.md exists but is not staged."""
    files = _staged_files(cwd)
    if not files:
        return None

    has_source = any(Path(f).suffix in _SOURCE_EXTENSIONS for f in files)
    if not has_source:
        return None

    changelog_staged = any(Path(f).name == "CHANGELOG.md" for f in files)
    if changelog_staged:
        return None

    changelog_exists = (Path(cwd) / "CHANGELOG.md").exists()
    if not changelog_exists:
        return None

    return {
        "decision": "warn",
        "reason": (
            "🟡 cairn: Source files staged but CHANGELOG.md is not.\n\n"
            "Run /cairn-changelog to add an entry, then stage it.\n\n"
            "To bypass: append # cairn:skip"
        ),
    }


def evaluate(command: str, cwd: str) -> dict | None:
    if "git commit" not in command:
        return None

    if is_bypassed(command, ("cairn:skip",)):
        return None

    parts = [
        (m.group(1) or m.group(2) or m.group(3) or "").strip()
        for m in _MSG_PATTERN.finditer(command)
    ]
    if not parts:
        return None

    message = "\n\n".join(p for p in parts if p)
    if not message:
        return None

    if len(message) < 10 or _WEAK.match(message):
        return {
            "decision": "deny",
            "reason": (
                f"🟡 cairn: Weak commit message '{message}'.\n\n"
                f"Run /cairn-commit to generate a message from the diff.\n\n"
                f"To bypass: append # cairn:skip"
            ),
        }

    return _changelog_warning(cwd)
