from dataclasses import asdict
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from claude_agent_sdk.types import SystemPromptPreset, ResultMessage
from loguru import logger

from core.error_processor import ErrorProcessor

# 常量
MAX_RETRIES = 3
provider_router = None


class LLMError(Exception):
    """LLM相关错误的异常类。"""

    pass


async def make_llm_api_call(
    messages: List[Dict[str, Any]],
    model_name: str,
    response_format: Optional[Any] = None,
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    stream: bool = True,  # 始终使用流式传输以获得更好的用户体验
    top_p: Optional[float] = None,
    model_id: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    system_prompt: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Union[Dict[str, Any], AsyncGenerator]:
    """使用Claude SDK进行语言模型API调用。"""
    logger.info(f"正在向模型发起LLM API调用: {model_name}，包含 {len(messages)} 条消息")

    try:
        options = ClaudeAgentOptions(
            system_prompt=SystemPromptPreset(
                type="preset",
                preset="claude_code",
                append="总是使用中文回复" if not system_prompt else system_prompt,
            ),
            include_partial_messages=True if stream else False,
            allowed_tools=["WebFetch", "WebSearch"],
        )
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)

            response = client.receive_response()

            if stream:
                return _wrap_streaming_response(response)

            async for chunk in response:
                if isinstance(chunk, ResultMessage):
                    return asdict(chunk)

    except Exception as e:
        # 使用ErrorProcessor一致地处理错误
        processed_error = ErrorProcessor.process_llm_error(
            e, context={"model": model_name}
        )
        ErrorProcessor.log_error(processed_error)
        raise LLMError(processed_error.message)


async def _wrap_streaming_response(response) -> ResultMessage:
    """包装流式响应以处理迭代过程中的错误。"""
    try:
        async for chunk in response:
            yield chunk
    except Exception as e:
        # 将流式错误转换为处理后的错误
        processed_error = ErrorProcessor.process_llm_error(e)
        ErrorProcessor.log_error(processed_error)
        raise LLMError(processed_error.message)
