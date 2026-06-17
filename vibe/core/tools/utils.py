from __future__ import annotations

import fnmatch
from pathlib import Path, PurePath

from vibe.core.scratchpad import is_scratchpad_path
from vibe.core.tools.base import ToolPermission
from vibe.core.tools.permissions import (
    PermissionContext,
    PermissionScope,
    RequiredPermission,
    _try_regex_match,
)


def _make_absolute(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def matches_pattern(text: str, pattern: str) -> bool:
    """Match text against a glob or regex pattern.

    Prefix 're:' triggers full regex search; otherwise uses fnmatch glob matching.
    Consistent with the 're:' prefix support in enabled_tools / disabled_tools.
    """
    if (m := _try_regex_match(text, pattern)) is not None:
        return m
    return fnmatch.fnmatch(text, pattern)


def resolve_path_permission(
    path_str: str, *, allowlist: list[str], denylist: list[str]
) -> PermissionContext | None:
    """Resolve permission for a file path against glob or regex patterns.

    Returns NEVER on denylist match, ALWAYS on allowlist match, None otherwise.
    Patterns prefixed with 're:' are matched as regexes; all others use fnmatch.
    """
    file_str = str(_make_absolute(path_str).resolve())

    for pattern in denylist:
        if matches_pattern(file_str, pattern):
            return PermissionContext(permission=ToolPermission.NEVER)

    for pattern in allowlist:
        if matches_pattern(file_str, pattern):
            return PermissionContext(permission=ToolPermission.ALWAYS)

    return None


def is_path_within_workdir(path_str: str) -> bool:
    """Return True if the resolved path is inside cwd or any project root.

    Project roots come from the harness manager (trusted cwd + ``--add-dir``).
    cwd is always in-bounds, even when the manager isn't initialized or when
    the project source is disabled.
    """
    try:
        resolved = _make_absolute(path_str).resolve()
    except (ValueError, OSError):
        return False
    if resolved.is_relative_to(Path.cwd().resolve()):
        return True
    from vibe.core.config.harness_files import get_harness_files_manager

    try:
        mgr = get_harness_files_manager()
    except RuntimeError:
        return False
    return any(resolved.is_relative_to(r) for r in mgr.project_roots)


def resolve_file_tool_permission(
    path_str: str,
    *,
    tool_name: str,
    allowlist: list[str],
    denylist: list[str],
    config_permission: ToolPermission,
    sensitive_patterns: list[str],
) -> PermissionContext | None:
    """Resolve permission for a file-based tool invocation.

    Checks scratchpad, then allowlist/denylist, then sensitive patterns, then workdir boundary.
    Returns PermissionContext with granular required_permissions when applicable.
    """
    if is_scratchpad_path(path_str):
        return PermissionContext(permission=ToolPermission.ALWAYS)

    if (
        result := resolve_path_permission(
            path_str, allowlist=allowlist, denylist=denylist
        )
    ) is not None:
        return result

    required: list[RequiredPermission] = []

    file_path = _make_absolute(path_str)
    file_str = str(file_path.resolve())

    for pattern in sensitive_patterns:
        if PurePath(file_str).match(pattern):
            required.append(
                RequiredPermission(
                    scope=PermissionScope.FILE_PATTERN,
                    invocation_pattern=file_path.name,
                    session_pattern="*",
                    label=f"accessing sensitive files ({tool_name})",
                )
            )
            break

    if not is_path_within_workdir(path_str):
        if config_permission == ToolPermission.NEVER:
            return PermissionContext(permission=ToolPermission.NEVER)
        resolved = file_path.resolve()
        parent_dir = str(resolved.parent)
        glob = str(Path(parent_dir) / "*")
        required.append(
            RequiredPermission(
                scope=PermissionScope.OUTSIDE_DIRECTORY,
                invocation_pattern=glob,
                session_pattern=glob,
                label=f"outside workdir ({glob})",
            )
        )

    if required:
        return PermissionContext(
            permission=ToolPermission.ASK, required_permissions=required
        )

    return None
