from collections.abc import AsyncGenerator
from dataclasses import asdict
from typing import Any, Optional, cast

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from claude_agent_sdk.types import McpServerConfig, PermissionMode, ResultMessage, SystemPromptPreset
from loguru import logger

from core.error_processor import ErrorProcessor

# 常量
MAX_RETRIES = 3
provider_router = None


class LLMError(Exception):
    """LLM相关错误的异常类。"""

    pass


async def make_llm_api_call(
    messages: list[dict[str, Any]],
    model_name: str,
    mcp_servers: Optional[dict[str, McpServerConfig]] = None,
    tools: Optional[list[dict[str, Any]]] = None,
    stream: bool = True,  # 始终使用流式传输以获得更好的用户体验
    permission_mode: PermissionMode | None = None,
    system_prompt: Optional[str] = None,
    prompt: Optional[str] = None,
    session_id: Optional[str] = None,
    output_format: dict[str, Any] | None = None,
) -> dict[str, Any] | AsyncGenerator | None:
    """使用Claude SDK进行语言模型API调用。"""
    logger.info(
        f"正在向 Claude Code 发起调用，模型: {model_name}，模式: {permission_mode}, 包含 {len(messages)} 条消息"
    )

    options = ClaudeAgentOptions(
        system_prompt=SystemPromptPreset(
            type="preset",
            preset="claude_code",
            append=system_prompt or "总是使用中文回复",
        ),
        include_partial_messages=True if stream else False,
        allowed_tools=["WebFetch", "WebSearch", "TodoWrite", "ExitPlanMode"],
        resume=session_id,
        permission_mode=permission_mode,
        output_format=output_format,
        mcp_servers=mcp_servers or {},
    )

    if stream:
        # 流式模式：返回一个生成器，在生成器内部管理 client 生命周期
        return _create_streaming_response(options, messages, prompt, model_name)

    # 非流式模式：在 async with 块内完成所有操作
    try:
        async with ClaudeSDKClient(options=options) as client:
            prompt_text = prompt or cast(str, messages[-1]["content"])
            await client.query(prompt_text)

            response = client.receive_response()
            async for chunk in response:
                if isinstance(chunk, ResultMessage):
                    return asdict(chunk)

    except Exception as e:
        # 使用ErrorProcessor一致地处理错误
        processed_error = ErrorProcessor.process_llm_error(e, context={"model": model_name})
        ErrorProcessor.log_error(processed_error)
        raise LLMError(processed_error.message)


async def _create_streaming_response(
    options: ClaudeAgentOptions,
    messages: list[dict[str, Any]],
    prompt: Optional[str] = None,
    model_name: Optional[str] = None,
) -> AsyncGenerator:
    """创建流式响应生成器，在生成器内部管理 client 生命周期。

    这确保了 client 在整个流式响应期间保持打开状态。
    """
    try:
        async with ClaudeSDKClient(options=options) as client:
            prompt_text = prompt or cast(str, messages[-1]["content"])
            await client.query(prompt_text)

            response = client.receive_response()

            async for chunk in response:
                yield chunk

    except Exception as e:
        # 将流式错误转换为处理后的错误
        processed_error = ErrorProcessor.process_llm_error(e, context={"model": model_name} if model_name else {})
        ErrorProcessor.log_error(processed_error)
        raise LLMError(processed_error.message)
