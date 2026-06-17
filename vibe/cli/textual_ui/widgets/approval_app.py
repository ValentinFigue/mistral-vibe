from __future__ import annotations

import time
from typing import ClassVar

from pydantic import BaseModel
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical, VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static

from vibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from vibe.cli.textual_ui.widgets.tool_widgets import get_approval_widget
from vibe.core.config import VibeConfig
from vibe.core.tools.permissions import RequiredPermission, ScopeOption

_INPUT_GRACE_PERIOD_S = 0.5


class ApprovalApp(Container):
    can_focus = True
    can_focus_children = False

    NUM_OPTIONS = 4

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("enter", "select", "Select", show=False),
        Binding("1", "select_1", "Yes", show=False),
        Binding("y", "select_1", "Yes", show=False),
        Binding("2", "select_2", "Allow this session", show=False),
        Binding("3", "select_3", "Always allow", show=False),
        Binding("4", "select_4", "No", show=False),
        Binding("n", "select_4", "No", show=False),
        Binding("s", "toggle_scope", "Switch scope", show=False),
    ]

    class ApprovalGranted(Message):
        def __init__(self, tool_name: str, tool_args: BaseModel) -> None:
            super().__init__()
            self.tool_name = tool_name
            self.tool_args = tool_args

    class ApprovalGrantedAlwaysTool(Message):
        def __init__(
            self,
            tool_name: str,
            tool_args: BaseModel,
            required_permissions: list[RequiredPermission],
        ) -> None:
            super().__init__()
            self.tool_name = tool_name
            self.tool_args = tool_args
            self.required_permissions = required_permissions

    class ApprovalGrantedAlwaysPermanent(Message):
        def __init__(
            self,
            tool_name: str,
            tool_args: BaseModel,
            required_permissions: list[RequiredPermission],
        ) -> None:
            super().__init__()
            self.tool_name = tool_name
            self.tool_args = tool_args
            self.required_permissions = required_permissions

    class ApprovalRejected(Message):
        def __init__(self, tool_name: str, tool_args: BaseModel) -> None:
            super().__init__()
            self.tool_name = tool_name
            self.tool_args = tool_args

    def __init__(
        self,
        tool_name: str,
        tool_args: BaseModel,
        config: VibeConfig,
        required_permissions: list[RequiredPermission] | None = None,
    ) -> None:
        super().__init__(id="approval-app")
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.config = config
        self.required_permissions = required_permissions or []
        self._scope_index: int = 0
        self.selected_option = 0
        self.content_container: Vertical | None = None
        self.title_widget = NoMarkupStatic(
            self._build_title(), classes="approval-title"
        )
        self.tool_info_container: Vertical | None = None
        self._scope_widget: NoMarkupStatic = NoMarkupStatic("", classes="approval-scope")
        self.option_widgets: list[Static] = []
        self.help_widget: Static | None = None
        self._mount_time: float = 0.0

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-content"):
            yield self.title_widget

            with VerticalScroll(classes="approval-tool-info-scroll"):
                self.tool_info_container = Vertical(
                    classes="approval-tool-info-container"
                )
                yield self.tool_info_container

        with Vertical(id="approval-options"):
            yield NoMarkupStatic("")
            yield self._scope_widget
            for _ in range(self.NUM_OPTIONS):
                widget = NoMarkupStatic("", classes="approval-option")
                self.option_widgets.append(widget)
                yield widget
            self.help_widget = NoMarkupStatic(
                "↑↓ navigate  Enter select  ESC reject", classes="approval-help"
            )
            yield self.help_widget

    def _build_title(self) -> str:
        if self.required_permissions:
            labels = ", ".join(rp.label for rp in self.required_permissions)
            return f"Permission for the {self.tool_name} tool ({labels})"
        return f"Permission for the {self.tool_name} tool"

    async def on_mount(self) -> None:
        self._mount_time = time.monotonic()
        await self._update_tool_info()
        rp_with_ladder = next((rp for rp in self.required_permissions if rp.scope_ladder), None)
        if rp_with_ladder:
            self._scope_index = rp_with_ladder.default_scope_index
        else:
            self._scope_index = min(1, len(self._get_ladder()) - 1)
        self._update_scope_widget()
        self._update_options()
        self.focus()
        self._recompute_height()
        self.screen.screen_layout_refresh_signal.subscribe(
            self, lambda _screen: self._recompute_height()
        )

    def _recompute_height(self) -> None:
        """Manual sizing: the scroll uses `1fr`, so `height: auto` cannot shrink to fit."""
        options = self.query_one("#approval-options", Vertical)
        scroll = self.query_one(".approval-tool-info-scroll", VerticalScroll)

        natural_height = (
            options.outer_size.height
            + self.title_widget.outer_size.height
            + scroll.virtual_size.height
            + self.gutter.height
        )

        # Cap the natural height if greater than max_height
        if max_height := self.styles.max_height:
            viewport = self.app.size
            parent_size = (
                self.parent.size if isinstance(self.parent, Widget) else viewport
            )
            resolved_max_height = int(max_height.resolve(parent_size, viewport))
            natural_height = min(natural_height, resolved_max_height)

        self.styles.height = natural_height

    def is_within_grace_period(self) -> bool:
        return (time.monotonic() - self._mount_time) < _INPUT_GRACE_PERIOD_S

    async def _update_tool_info(self) -> None:
        if not self.tool_info_container:
            return

        approval_widget = get_approval_widget(self.tool_name, self.tool_args)
        await self.tool_info_container.remove_children()
        await self.tool_info_container.mount(approval_widget)

    def _get_ladder(self) -> list[ScopeOption]:
        for rp in self.required_permissions:
            if rp.scope_ladder:
                return rp.scope_ladder
        if self.required_permissions:
            patterns = list(dict.fromkeys(
                rp.session_pattern for rp in self.required_permissions if rp.session_pattern != "*"
            ))
            if patterns:
                return [
                    ScopeOption(label=f"Pattern: {patterns[0]}", pattern=patterns[0]),
                    ScopeOption(label=f"Full tool: {self.tool_name}", pattern=None),
                ]
        return [ScopeOption(label=f"Full tool: {self.tool_name}", pattern=None)]

    def _selected_rung(self) -> ScopeOption:
        return self._get_ladder()[self._scope_index]

    def _get_active_permissions(self) -> list[RequiredPermission]:
        rung = self._selected_rung()
        if rung.pattern is None:
            return []  # full-tool → approve_always([]) sets tool ALWAYS
        return [
            rp.model_copy(update={"session_pattern": rung.pattern})
            for rp in self.required_permissions
        ]

    def _scope_hint_text(self) -> str:
        if not self.required_permissions:
            return ""
        rung = self._selected_rung()
        if rung.pattern is None:
            return f"  Full tool: {self.tool_name}"
        return f"  {rung.pattern}"

    def _update_scope_widget(self) -> None:
        if not self.required_permissions:
            self._scope_widget.update("")
            return
        ladder = self._get_ladder()
        if len(ladder) <= 1:
            self._scope_widget.update("")
            return
        rung = ladder[self._scope_index]
        self._scope_widget.update(f"  Scope: {rung.label}   [s to change]")

    def action_toggle_scope(self) -> None:
        if not self.required_permissions:
            return
        self._scope_index = (self._scope_index + 1) % len(self._get_ladder())
        self._update_scope_widget()
        self._update_options()
        self._recompute_height()

    def _update_options(self) -> None:
        hint = self._scope_hint_text()
        # Option 1 (Allow once) posts no rule, so scope is irrelevant — no hint shown.
        suffix = f"\n{hint}" if hint else ""
        options = [
            ("Allow once", "yes"),
            (f"Allow for remainder of this session{suffix}", "yes"),
            (f"Always allow (saves to config){suffix}", "yes"),
            ("Deny", "no"),
        ]

        for idx, ((text, color_type), widget) in enumerate(
            zip(options, self.option_widgets, strict=True)
        ):
            is_selected = idx == self.selected_option

            cursor = "› " if is_selected else "  "
            option_text = f"{cursor}{idx + 1}. {text}"

            widget.update(option_text)

            widget.remove_class("approval-cursor-selected")
            widget.remove_class("approval-option-selected")
            widget.remove_class("approval-option-yes")
            widget.remove_class("approval-option-no")

            if is_selected:
                widget.add_class("approval-cursor-selected")
                if color_type == "yes":
                    widget.add_class("approval-option-yes")
                else:
                    widget.add_class("approval-option-no")
            else:
                widget.add_class("approval-option-selected")
                if color_type == "yes":
                    widget.add_class("approval-option-yes")
                else:
                    widget.add_class("approval-option-no")

    def action_move_up(self) -> None:
        self.selected_option = (self.selected_option - 1) % self.NUM_OPTIONS
        self._update_options()

    def action_move_down(self) -> None:
        self.selected_option = (self.selected_option + 1) % self.NUM_OPTIONS
        self._update_options()

    def _select_if_unguarded(self, option: int) -> None:
        if self.is_within_grace_period():
            return
        self.selected_option = option
        self._handle_selection(option)

    def action_select(self) -> None:
        self._select_if_unguarded(self.selected_option)

    def action_select_1(self) -> None:
        self._select_if_unguarded(0)

    def action_select_2(self) -> None:
        self._select_if_unguarded(1)

    def action_select_3(self) -> None:
        self._select_if_unguarded(2)

    def action_select_4(self) -> None:
        self._select_if_unguarded(3)

    def action_reject(self) -> None:
        self._select_if_unguarded(3)

    def _handle_selection(self, option: int) -> None:
        match option:
            case 0:
                self.post_message(
                    self.ApprovalGranted(
                        tool_name=self.tool_name, tool_args=self.tool_args
                    )
                )
            case 1:
                self.post_message(
                    self.ApprovalGrantedAlwaysTool(
                        tool_name=self.tool_name,
                        tool_args=self.tool_args,
                        required_permissions=self._get_active_permissions(),
                    )
                )
            case 2:
                self.post_message(
                    self.ApprovalGrantedAlwaysPermanent(
                        tool_name=self.tool_name,
                        tool_args=self.tool_args,
                        required_permissions=self._get_active_permissions(),
                    )
                )
            case 3:
                self.post_message(
                    self.ApprovalRejected(
                        tool_name=self.tool_name, tool_args=self.tool_args
                    )
                )

    def on_blur(self, event: events.Blur) -> None:
        self.call_after_refresh(self.focus)
