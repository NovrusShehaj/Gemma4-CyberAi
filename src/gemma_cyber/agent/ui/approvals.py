"""Interactive approval prompts (plan §10).

The approver is consulted *after* `PermissionGuard` has already allowed a call,
so a prompt can only ever decline something legal — it can never authorise
something the mode forbids. That ordering is why `-y/--yes` is safe: it is an
`Approver`, not a permission.
"""

from __future__ import annotations

import asyncio

from gemma_cyber.agent.sanitize import sanitize_for_terminal
from gemma_cyber.agent.tools.base import ApprovalRequest
from gemma_cyber.agent.types import SideEffect
from gemma_cyber.agent.ui.render import Renderer

__all__ = ["TtyApprover"]

_PROMPT_HINT = {
    SideEffect.WORKSPACE_WRITE: "apply this change?",
    SideEffect.PROCESS: "run this command?",
}


class TtyApprover:
    """Asks the user. Anything that is not an explicit yes is a no."""

    def __init__(self, renderer: Renderer, *, remember: bool = True) -> None:
        self.renderer = renderer
        self.remember = remember
        #: Tool names the user chose to stop being asked about, this session only.
        self._session_yes: set[str] = set()

    async def approve(self, request: ApprovalRequest) -> bool:
        if request.tool in self._session_yes:
            return True

        self.renderer.end_stream()
        self.renderer.rule("approval required")
        self.renderer.plain(request.summary)
        self.renderer.info(request.decision.reason)
        if request.preview:
            self.renderer.diff(request.preview)

        hint = _PROMPT_HINT.get(request.side_effect, "allow this?")
        suffix = " [y/N/a=always this tool]" if self.remember else " [y/N]"
        answer = await asyncio.to_thread(self._ask, f"{hint}{suffix} ")
        choice = sanitize_for_terminal(answer).strip().lower()

        if choice in ("a", "always") and self.remember:
            # Session-scoped only. There is deliberately no "remember forever"
            # for shell.exec (plan §15).
            if request.side_effect is not SideEffect.PROCESS:
                self._session_yes.add(request.tool)
            return True
        return choice in ("y", "yes")

    @staticmethod
    def _ask(prompt: str) -> str:
        try:
            return input(prompt)
        except (EOFError, KeyboardInterrupt):
            # No answer is a refusal. Fail closed.
            return "n"
