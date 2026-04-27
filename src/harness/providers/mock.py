# src/harness/providers/mock.py
from __future__ import annotations

from .base import Provider, ProviderResponse


class MockProvider(Provider):
    """A scripted provider for teaching and testing.

    Walks through a fixed list of responses, one per call.
    """

    def __init__(self, responses: list[ProviderResponse]) -> None:
        self._responses = list(responses)
        self._index = 0

    def complete(self, transcript: list[dict], tools: list[dict]) -> ProviderResponse:
        if self._index >= len(self._responses):
            raise RuntimeError("mock ran out of responses")
        response = self._responses[self._index]
        self._index += 1
        return response
