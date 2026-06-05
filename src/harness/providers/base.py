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
    reasoning_text: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0 

    @property
    def is_tool_call(self) -> bool:
        return self.tool_name is not None
    

    @property
    def is_final(self)->bool:
        return self.text is not None and self.tool_name is None


class Provider:
    def complete(self, transcript:list[dict],tools:list[dict])->ProviderResponse:
        """given the transcript and available tools.    """