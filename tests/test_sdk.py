import pytest
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from claude_agent_sdk.types import SystemPromptPreset


async def messages_to_async_iterable(messages):
    """将消息列表转换为异步可迭代对象"""
    for msg in messages:
        yield msg


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
        history = [
            {"role": "user", "content": "你好，你是谁？"},
            {"role": "assistant", "content": "我是小明"},
        ]
        new_input = "你是谁？"

        messages = history + [{"role": "user", "content": new_input}]
        # 将列表转换为异步可迭代对象
        await client.query(messages_to_async_iterable(messages))

        async for message in client.receive_response():
            print(message)
