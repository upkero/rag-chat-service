from abc import ABC, abstractmethod
from collections.abc import Sequence

from src.app.contracts.llm.llm_message import LLMMessage
from src.app.contracts.llm.llm_response import LLMResponse


class LLMClient(ABC):
    @abstractmethod
    async def complete(self, messages: Sequence[LLMMessage]) -> LLMResponse: ...

    @abstractmethod
    async def ping(self) -> bool: ...

    @abstractmethod
    async def close(self) -> None: ...
