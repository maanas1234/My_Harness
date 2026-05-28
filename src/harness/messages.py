from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4


Role = Literal["user","assistant","system"]

@dataclass(frozen=True)
class TextBlock:
    text: str
    kind : Literal["text"] = "text"

@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict
    kind: Literal["tool_call"] = "tool_call"

@dataclass(frozen=True)
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False
    kind: Literal["tool_result"] = "tool_result"


@dataclass(frozen=True)
class ReasoningBlock:
    text: str
    metadata: dict = field(default_factory=dict)
    kind: Literal["reasoning"] = "reasoning"


Block = TextBlock | ToolCall | ToolResult | ReasoningBlock


@dataclass
class Message:
    role: Role
    blocks : list[Block]
    created_at: datetime = field(default_factory=lambda:datetime.now(timezone.utc))
    id: str = field(default_factory=lambda: str(uuid4()))

    @classmethod
    def user_text(cls, text:str) -> "Message":
        return cls(role ="user",blocks = [TextBlock(text)])
    
    @classmethod
    def assistant_text(cls, text: str, *, reasoning: ReasoningBlock | None = None) ->"Message":
        blocks: list[Block] = []
        if reasoning is not None:
            blocks.append(reasoning)
        blocks.append(TextBlock(text))
        return cls(role="assistant", blocks=blocks)
    
    

    @classmethod
    def assistant_tool_call(cls, call: ToolCall, *, reasoning: ReasoningBlock | None = None) ->"Message":
        blocks: list[Block] = []
        if reasoning is not None:
            blocks.append(reasoning)
        blocks.append(call)
        return cls(role="assistant",blocks =blocks)
    

    @classmethod
    def tool_result(cls, result: ToolResult) -> "Message":
        return cls(role="user", blocks=[result])
    

    @classmethod
    def from_assistant_response(cls, response) -> "Message":
        reasoning = None
        has_reasoning = (
        bool(response.reasoning_text)
        or bool(getattr(response, "reasoning_metadata", None))
        )
        if has_reasoning:
            meta: dict = {"provider_tokens": response.reasoning_tokens}
            meta.update(getattr(response, "reasoning_metadata", None) or {})
            reasoning = ReasoningBlock(
            text=response.reasoning_text or "",
            metadata=meta,
            )
        blocks: list[Block] = []
        if reasoning is not None:
            blocks.append(reasoning)
        if response.tool_calls:
        # One assistant message carries every ToolCall block from
        # this turn — both providers accept multi-tool_use messages
        # on round-trip.
            for ref in response.tool_calls:
                blocks.append(ToolCall(id=ref.id, name=ref.name, args=dict(ref.args)))
        else:
            blocks.append(TextBlock(text=response.text or ""))
        return cls(role="assistant", blocks=blocks)

@dataclass
class Transcript:
    messages: list[Message] = field(default_factory=list)
    system: str | None = None
    def append(self, message: Message) -> None:
        self.messages.append(message)
    def extend(self, messages: list[Message]) -> None:
        self.messages.extend(messages)
    def last(self) -> Message | None:
        return self.messages[-1] if self.messages else None
    def __len__(self) -> int:
        return len(self.messages)