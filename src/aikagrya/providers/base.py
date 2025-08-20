from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ChatProvider(ABC):
    """Abstract chat provider interface."""

    @abstractmethod
    def send(self, messages: List[Dict[str, str]], *, model: str, max_tokens: int = 1024, temperature: float = 0.7) -> Dict[str, Any]:
        """Send chat messages and return provider-normalized response.

        The return should include at minimum keys: {"content": str, "finish_reason": str, "usage": dict}.
        """
        raise NotImplementedError


