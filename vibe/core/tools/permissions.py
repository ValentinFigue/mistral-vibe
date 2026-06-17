from __future__ import annotations

import asyncio
from enum import StrEnum, auto
import fnmatch
import re as _re

from pydantic import BaseModel, Field

from vibe.core.tools.base import ToolPermission


class PermissionScope(StrEnum):
    COMMAND_PATTERN = auto()
    OUTSIDE_DIRECTORY = auto()
    FILE_PATTERN = auto()
    URL_PATTERN = auto()


class ScopeOption(BaseModel):
    label: str
    pattern: str | None  # None = full-tool sentinel → approve_always([]) → tool ALWAYS


class RequiredPermission(BaseModel):
    scope: PermissionScope
    invocation_pattern: str
    session_pattern: str
    label: str
    scope_ladder: list[ScopeOption] = Field(default_factory=list)
    default_scope_index: int = 0


class PermissionContext(BaseModel):
    permission: ToolPermission
    required_permissions: list[RequiredPermission] = Field(default_factory=list)
    reason: str | None = None


class ApprovedRule(BaseModel):
    tool_name: str
    scope: PermissionScope
    session_pattern: str


def _try_regex_match(text: str, pattern: str) -> bool | None:
    """If pattern starts with 're:', run a regex search and return bool.

    Returns None for non-re: patterns so the caller can apply its own fallback.
    """
    if not pattern.startswith("re:"):
        return None
    return bool(_re.search(pattern[3:], text))


def wildcard_match(text: str, pattern: str) -> bool:
    """Match text against a session rule pattern.

    Prefix 're:' triggers regex search. Otherwise uses fnmatch glob matching
    with one extension: if the pattern ends with ' *', trailing args are optional
    (the pattern matches both with and without trailing arguments).
    """
    if (m := _try_regex_match(text, pattern)) is not None:
        return m
    if fnmatch.fnmatch(text, pattern):
        return True
    if pattern.endswith(" *") and fnmatch.fnmatch(text, pattern[:-2]):
        return True
    return False


class PermissionStore:
    def __init__(self) -> None:
        self._rules: list[ApprovedRule] = []
        self._tool_permissions: dict[str, ToolPermission] = {}
        self.lock = asyncio.Lock()

    def add_rule(self, rule: ApprovedRule) -> None:
        self._rules.append(rule)

    def covers(self, tool_name: str, rp: RequiredPermission) -> bool:
        return any(
            rule.tool_name == tool_name
            and rule.scope == rp.scope
            and wildcard_match(rp.invocation_pattern, rule.session_pattern)
            for rule in self._rules
        )

    def set_tool_permission(self, tool_name: str, permission: ToolPermission) -> None:
        self._tool_permissions[tool_name] = permission

    def get_tool_permission(self, tool_name: str) -> ToolPermission | None:
        return self._tool_permissions.get(tool_name)
