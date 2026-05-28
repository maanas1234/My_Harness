from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Protocol
from uuid import uuid4
import os


# =========================================================
# MESSAGE / BLOCK SYSTEM
# =========================================================

Role = Literal["user", "assistant", "system"]


@dataclass(frozen=True)
class TextBlock:
    text: str
    kind: Literal["text"] = "text"


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
    blocks: list[Block]

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    id: str = field(default_factory=lambda: str(uuid4()))

    @classmethod
    def user_text(cls, text: str) -> "Message":
        return cls(
            role="user",
            blocks=[TextBlock(text=text)],
        )

    @classmethod
    def assistant_text(
        cls,
        text: str,
        *,
        reasoning: ReasoningBlock | None = None,
    ) -> "Message":

        blocks: list[Block] = []

        if reasoning is not None:
            blocks.append(reasoning)

        blocks.append(TextBlock(text=text))

        return cls(
            role="assistant",
            blocks=blocks,
        )

    @classmethod
    def assistant_tool_call(
        cls,
        call: ToolCall,
        *,
        reasoning: ReasoningBlock | None = None,
    ) -> "Message":

        blocks: list[Block] = []

        if reasoning is not None:
            blocks.append(reasoning)

        blocks.append(call)

        return cls(
            role="assistant",
            blocks=blocks,
        )

    @classmethod
    def tool_result(cls, result: ToolResult) -> "Message":
        return cls(
            role="user",
            blocks=[result],
        )

    @classmethod
    def from_assistant_response(
        cls,
        response: "ProviderResponse",
    ) -> "Message":

        blocks: list[Block] = []

        # -------------------------------------------------
        # reasoning block
        # -------------------------------------------------

        if response.reasoning_text is not None:
            reasoning = ReasoningBlock(
                text=response.reasoning_text
            )
            blocks.append(reasoning)

        # -------------------------------------------------
        # tool call
        # -------------------------------------------------

        if response.is_tool_call:

            tool_call = ToolCall(
                id=response.tool_call_id or str(uuid4()),
                name=response.tool_name or "",
                args=response.tool_args or {},
            )

            blocks.append(tool_call)

        # -------------------------------------------------
        # final text
        # -------------------------------------------------

        elif response.is_final:

            blocks.append(
                TextBlock(
                    text=response.text or ""
                )
            )

        return cls(
            role="assistant",
            blocks=blocks,
        )


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


# =========================================================
# PROVIDER RESPONSE
# =========================================================

@dataclass(frozen=True)
class ProviderResponse:
    """
    Normalized provider response.

    ALL providers should convert their native responses
    into this format.
    """

    text: str | None = None

    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_args: dict | None = None

    reasoning_text: str | None = None

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def is_tool_call(self) -> bool:
        return self.tool_name is not None

    @property
    def is_final(self) -> bool:
        return (
            self.text is not None
            and self.tool_name is None
        )


class Provider(Protocol):

    name: str

    def complete(
        self,
        transcript: Transcript,
        tools: list[dict],
    ) -> ProviderResponse:
        ...


# =========================================================
# ANTHROPIC PROVIDER
# =========================================================

class AnthropicProvider(Provider):

    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-4-0",
        client: Any | None = None,
        enable_thinking: bool = False,
        thinking_budget_tokens: int = 2000,
        max_tokens: int = 4096,
    ) -> None:

        self.model = model
        self.enable_thinking = enable_thinking
        self.thinking_budget_tokens = thinking_budget_tokens
        self.max_tokens = max_tokens

        if client is None:

            from anthropic import Anthropic

            client = Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )

        self._client = client

    # =====================================================
    # MAIN INFERENCE METHOD
    # =====================================================

    def complete(
        self,
        transcript: Transcript,
        tools: list[dict],
    ) -> ProviderResponse:

        kwargs: dict[str, Any] = {

            "model": self.model,

            "max_tokens": self.max_tokens,

            "messages": [
                _to_anthropic(
                    m,
                    self.enable_thinking,
                )
                for m in transcript.messages
            ],

            "tools": tools,
        }

        if transcript.system:
            kwargs["system"] = transcript.system

        if self.enable_thinking:

            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget_tokens,
            }

        # -----------------------------------------------
        # REAL API CALL
        # -----------------------------------------------

        raw = self._client.messages.create(**kwargs)

        # -----------------------------------------------
        # convert provider response
        # into normalized response
        # -----------------------------------------------

        return _from_anthropic(raw)


# =========================================================
# INTERNAL -> ANTHROPIC
# =========================================================

def _to_anthropic(
    message: Message,
    keep_reasoning: bool,
) -> dict:

    content: list[dict] = []

    for block in message.blocks:

        if (
            isinstance(block, ReasoningBlock)
            and not keep_reasoning
        ):
            continue

        content.append(
            _block_to_anthropic(block)
        )

    return {
        "role": message.role,
        "content": content,
    }


def _block_to_anthropic(block: Block) -> dict:

    match block:

        # ---------------------------------------------
        # text
        # ---------------------------------------------

        case TextBlock(text=t):

            return {
                "type": "text",
                "text": t,
            }

        # ---------------------------------------------
        # tool call
        # ---------------------------------------------

        case ToolCall(id=i, name=n, args=a):

            return {
                "type": "tool_use",
                "id": i,
                "name": n,
                "input": a,
            }

        # ---------------------------------------------
        # tool result
        # ---------------------------------------------

        case ToolResult(
            call_id=i,
            content=c,
            is_error=err,
        ):

            return {
                "type": "tool_result",
                "tool_use_id": i,
                "content": c,
                "is_error": err,
            }

        # ---------------------------------------------
        # reasoning
        # ---------------------------------------------

        case ReasoningBlock(
            text=t,
            metadata=meta,
        ):

            out = {
                "type": "thinking",
                "thinking": t,
            }

            if (
                sig := meta.get("signature")
            ) is not None:

                out["signature"] = sig

            return out

    raise ValueError(
        f"Unhandled block type: {type(block)}"
    )


# =========================================================
# ANTHROPIC -> INTERNAL
# =========================================================

def _from_anthropic(
    raw: Any,
) -> ProviderResponse:

    # -----------------------------------------------------
    # collect reasoning
    # -----------------------------------------------------

    thinking_texts = [

        b.thinking

        for b in raw.content

        if b.type == "thinking"
    ]

    reasoning_text = (
        "\n".join(thinking_texts)
        if thinking_texts
        else None
    )

    # -----------------------------------------------------
    # tool call path
    # -----------------------------------------------------

    for block in raw.content:

        if block.type == "tool_use":

            return ProviderResponse(

                tool_call_id=block.id,

                tool_name=block.name,

                tool_args=dict(block.input),

                reasoning_text=reasoning_text,

                input_tokens=raw.usage.input_tokens,

                output_tokens=raw.usage.output_tokens,
            )

    # -----------------------------------------------------
    # final text path
    # -----------------------------------------------------

    texts = [

        b.text

        for b in raw.content

        if b.type == "text"
    ]

    return ProviderResponse(

        text="\n".join(texts),

        reasoning_text=reasoning_text,

        input_tokens=raw.usage.input_tokens,

        output_tokens=raw.usage.output_tokens,
    )