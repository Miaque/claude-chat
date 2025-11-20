import pytest
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from claude_agent_sdk.types import SystemPromptPreset


@pytest.mark.asyncio
async def test_sdk():
    options = ClaudeAgentOptions(
        system_prompt=SystemPromptPreset(
            type="preset", preset="claude_code", append="总是使用中文回复"
        ),
        include_partial_messages=False,
        allowed_tools=["WebFetch", "WebSearch"],
    )
    async with ClaudeSDKClient(options=options) as client:
        await client.query("今天星期几")

        print("\n")
        async for message in client.receive_response():
            print(message)