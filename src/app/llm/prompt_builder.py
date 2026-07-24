from collections.abc import Iterable

from src.app.contracts.llm.llm_message import LLMMessage, LLMRole


class PromptBuilder:
    """Small fluent builder for assembling ordered LLM message lists (DRY helper)."""

    def __init__(self) -> None:
        self._messages: list[LLMMessage] = []

    def add(self, *, role: LLMRole, content: str) -> "PromptBuilder":
        self._messages.append(LLMMessage(role=role, content=content))
        return self

    def system(self, content: str) -> "PromptBuilder":
        return self.add(role="system", content=content)

    def user(self, content: str) -> "PromptBuilder":
        return self.add(role="user", content=content)

    def assistant(self, content: str) -> "PromptBuilder":
        return self.add(role="assistant", content=content)

    def extend(self, messages: Iterable[LLMMessage]) -> "PromptBuilder":
        self._messages.extend(messages)
        return self

    def build(self) -> list[LLMMessage]:
        return list(self._messages)
