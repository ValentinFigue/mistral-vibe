"""Full-screen interactive view of the current workflow graph (bottom panel).

A navigable `OptionList` of nodes (coloured by run state) beside a live per-node details
pane. Opened by `/graph` and `/blocks show <name>`; dismissed with escape. The widget is
"dumb" — it renders a list of :class:`NodeView` the app computes from the graph + cache, so
it never imports the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from vibe.cli.clipboard import copy_text_to_clipboard
from vibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from vibe.core.graph import render

_HELP = "↑↓ Navigate  ·  c Copy Mermaid  ·  Esc Close"


@dataclass
class NodeView:
    """Everything the panel needs to render one node (plain data, no engine types)."""

    id: str
    op: str
    inputs: dict[str, str] = field(default_factory=dict)
    params: dict[str, object] = field(default_factory=dict)
    state: str | None = None
    cached: bool | None = None
    output: str | None = None


class GraphApp(Container):
    can_focus_children = True
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Close", show=False),
        Binding("q", "close", "Close", show=False),
        Binding("c", "copy_mermaid", "Copy Mermaid", show=False),
    ]

    DEFAULT_CSS = """
    GraphApp { height: auto; max-height: 80%; }
    GraphApp #graph-content { height: auto; }
    GraphApp #graph-body { height: auto; }
    GraphApp #graph-options { width: 45%; height: auto; max-height: 24; }
    GraphApp #graph-detail { width: 55%; padding: 0 1; }
    """

    class Closed(Message):
        pass

    def __init__(self, title: str, nodes: list[NodeView], mermaid: str = "") -> None:
        super().__init__(id="graph-app")
        self._title = title
        self._views = {node.id: node for node in nodes}
        self._order = [node.id for node in nodes]
        self._mermaid = mermaid
        self.detail_text = ""  # plain text of the current details pane (for tests/introspection)

    def compose(self) -> ComposeResult:
        with Vertical(id="graph-content"):
            yield NoMarkupStatic(self._title, id="graph-title", classes="settings-title")
            if not self._order:
                yield NoMarkupStatic("No workflow graph in this session yet.")
                yield NoMarkupStatic(_HELP, id="graph-help", classes="settings-help")
                return
            with Horizontal(id="graph-body"):
                options = [
                    Option(
                        render.node_line(n.id, n.op, n.inputs, n.state, n.cached), id=n.id
                    )
                    for n in (self._views[i] for i in self._order)
                ]
                yield OptionList(*options, id="graph-options")
                yield Static(id="graph-detail")
            yield NoMarkupStatic(_HELP, id="graph-help", classes="settings-help")

    def on_mount(self) -> None:
        if self._order:
            self.query_one(OptionList).focus()
            self._show_detail(self._order[0])

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        if event.option.id:
            self._show_detail(event.option.id)

    def _show_detail(self, node_id: str) -> None:
        node = self._views.get(node_id)
        if node is None:
            return
        detail = render.node_detail(
            node.id, node.op, node.inputs, node.params, node.state, node.cached, node.output
        )
        self.detail_text = str(detail)
        self.query_one("#graph-detail", Static).update(detail)

    def action_close(self) -> None:
        self.post_message(self.Closed())

    def action_copy_mermaid(self) -> None:
        if self._mermaid:
            copy_text_to_clipboard(
                self.app, self._mermaid, success_message="Mermaid diagram copied to clipboard"
            )
