from harness.agent import run
from harness.providers.base import ProviderResponse
from harness.providers.mock import MockProvider

def calc(expression: str) -> str:
# dangerous in real life; fine for a mock
    return str(eval(expression, {"__builtins__": {}}, {}))


mock = MockProvider([
ProviderResponse(
kind="tool_call",
tool_name="calc",
tool_args={"expression": "2 + 2"},
tool_call_id="call-1"),
ProviderResponse(kind="text", text="2 + 2 is 4."),
])


tool_schemas = [{
"name": "calc",
"description": "Evaluate a Python arithmetic expression.",
"input_schema": {
"type": "object",
"properties": {"expression": {"type": "string"}},
"required": ["expression"],
},
}]

answer = run(
provider=mock,
tools={"calc": calc},
tool_schemas=tool_schemas,
user_message="What is 2 + 2?",
)

print(answer)