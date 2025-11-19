from typing import Any, AsyncGenerator, Dict, List, Optional, Union
from loguru import logger

# Constants
MAX_RETRIES = 3
provider_router = None


class LLMError(Exception):
    """Exception for LLM-related errors."""

    pass


def _add_tools_config(
    params: Dict[str, Any], tools: Optional[List[Dict[str, Any]]], tool_choice: str
) -> None:
    """Add tools configuration to parameters."""
    if tools is None:
        return

    params.update({"tools": tools, "tool_choice": tool_choice})
    # logger.debug(f"Added {len(tools)} tools to API parameters")


async def make_llm_api_call(
    messages: List[Dict[str, Any]],
    model_name: str,
    response_format: Optional[Any] = None,
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    stream: bool = True,  # Always stream for better UX
    top_p: Optional[float] = None,
    model_id: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Union[Dict[str, Any], AsyncGenerator, ModelResponse]:
    """Make an API call to a language model using LiteLLM."""
    logger.info(
        f"Making LLM API call to model: {model_name} with {len(messages)} messages"
    )

    # Prepare parameters using centralized model configuration
    from core.ai_models import model_manager

    resolved_model_name = model_manager.resolve_model_id(model_name)
    # logger.debug(f"Model resolution: '{model_name}' -> '{resolved_model_name}'")

    # Only pass headers/extra_headers if they are not None to avoid overriding model config
    override_params = {
        "messages": messages,
        "temperature": temperature,
        "response_format": response_format,
        "top_p": top_p,
        "stream": stream,
        "api_key": api_key,
        "api_base": api_base,
    }

    # Only add headers if they are provided (not None)
    if headers is not None:
        override_params["headers"] = headers
    if extra_headers is not None:
        override_params["extra_headers"] = extra_headers

    params = model_manager.get_litellm_params(resolved_model_name, **override_params)

    # logger.debug(f"Parameters from model_manager.get_litellm_params: {params}")

    if model_id:
        params["model_id"] = model_id

    if stream:
        params["stream_options"] = {"include_usage": True}

    _add_tools_config(params, tools, tool_choice)

    try:
        response = await provider_router.acompletion(**params)

        # For streaming responses, we need to handle errors that occur during iteration
        if hasattr(response, "__aiter__") and stream:
            return _wrap_streaming_response(response)

        return response

    except Exception as e:
        # Use ErrorProcessor to handle the error consistently
        processed_error = ErrorProcessor.process_llm_error(
            e, context={"model": model_name}
        )
        ErrorProcessor.log_error(processed_error)
        raise LLMError(processed_error.message)


async def _wrap_streaming_response(response) -> AsyncGenerator:
    """Wrap streaming response to handle errors during iteration."""
    try:
        async for chunk in response:
            yield chunk
    except Exception as e:
        # Convert streaming errors to processed errors
        processed_error = ErrorProcessor.process_llm_error(e)
        ErrorProcessor.log_error(processed_error)
        raise LLMError(processed_error.message)
