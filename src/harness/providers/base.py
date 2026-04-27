from __future__  import annotations
from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True)
class ProviderResponse:
    """what the provider gives us back"""
    kind: str  # "text" or "tool_call"
    text: str | None = None
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_call_id: str | None = None


class provider:
    def complete(self, transcript:list[dict],tools:list[dict])->ProviderResponse:
        """given the transcript and available tools.    """