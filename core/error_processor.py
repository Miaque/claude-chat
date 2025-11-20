from dataclasses import dataclass
from typing import Any, Dict, Optional

from claude_agent_sdk import (
    ClaudeSDKError,
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    ProcessError,
)
from loguru import logger


@dataclass
class ProcessedError:
    """标准化的错误表示。"""

    error_type: str
    message: str
    original_error: Optional[Exception] = None
    context: Optional[Dict[str, Any]] = None

    def to_stream_dict(self) -> Dict[str, Any]:
        """转换为流兼容的错误字典。"""
        return {
            "type": "status",
            "status": "error",
            "message": self.message,
            "error_type": self.error_type,
        }


class ErrorProcessor:
    """使用 LiteLLM 的异常类型处理 LLM 相关错误。"""

    @staticmethod
    def process_llm_error(
        error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> ProcessedError:
        """Process LLM-related errors using LiteLLM's exception types."""
        error_message = ErrorProcessor.safe_error_to_string(error)

        if isinstance(error, CLIJSONDecodeError):
            return ProcessedError(
                error_type="cli_json_decode_error",
                message=f"JSON 解码错误：{error_message}",
                original_error=error,
                context=context,
            )

        elif isinstance(error, ProcessError):
            return ProcessedError(
                error_type="process_error",
                message=f"进程错误：{error_message}",
                original_error=error,
                context=context,
            )

        elif isinstance(error, CLIConnectionError):
            return ProcessedError(
                error_type="cli_connection_error",
                message=f"CLI 连接错误：{error_message}",
                original_error=error,
                context=context,
            )

        elif isinstance(error, CLINotFoundError):
            return ProcessedError(
                error_type="cli_not_found_error",
                message=f"CLI 未找到：{error_message}",
                original_error=error,
                context=context,
            )

        elif isinstance(error, ClaudeSDKError):
            return ProcessedError(
                error_type="claude_sdk_error",
                message=f"Claude SDK 错误：{error_message}",
                original_error=error,
                context=context,
            )

        else:
            # 未知错误类型的回退处理
            return ProcessedError(
                error_type="llm_error",
                message=f"LLM 错误：{error_message}",
                original_error=error,
                context=context,
            )

    @staticmethod
    def process_tool_error(
        error: Exception, tool_name: str, context: Optional[Dict[str, Any]] = None
    ) -> ProcessedError:
        """处理工具执行错误。"""
        return ProcessedError(
            error_type="tool_execution_error",
            message=f"工具 '{tool_name}' 执行失败：{ErrorProcessor.safe_error_to_string(error)}",
            original_error=error,
            context=context,
        )

    @staticmethod
    def process_system_error(
        error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> ProcessedError:
        """处理通用系统错误。"""
        return ProcessedError(
            error_type="system_error",
            message=f"System error: {ErrorProcessor.safe_error_to_string(error)}",
            original_error=error,
            context=context,
        )

    @staticmethod
    def safe_error_to_string(error: Exception) -> str:
        """安全地将异常转换为字符串，并带有回退机制"""
        try:
            return str(error)

        except Exception:
            try:
                # 处理 error.args[0] 可能是列表或其他非字符串类型的情况
                if error.args:
                    first_arg = error.args[0]
                    if isinstance(first_arg, (list, tuple)):
                        # 安全地将列表/元组转换为字符串
                        return f"{type(error).__name__}: {str(first_arg)}"
                    else:
                        return f"{type(error).__name__}: {str(first_arg)}"
                else:
                    return f"{type(error).__name__}: 未知错误"
            except Exception:
                return f"Error of type {type(error).__name__}"

    @staticmethod
    def log_error(processed_error: ProcessedError, level: str = "error") -> None:
        """使用适当的级别记录已处理的错误。"""
        log_func = getattr(logger, level, logger.error)

        log_message = (
            f"[{processed_error.error_type.upper()}] {processed_error.message}"
        )

        # 永远不要将 exc_info 传递给 structlog - 它会导致复杂异常的连接错误
        # 相反，安全地记录错误详情
        if processed_error.original_error:
            try:
                error_details = f"原始错误: {ErrorProcessor.safe_error_to_string(processed_error.original_error)}"
                log_func(f"{log_message} | {error_details}")
            except Exception:
                # If even our safe conversion fails, just log the message
                log_func(log_message)
        else:
            log_func(log_message)

        if processed_error.context:
            logger.debug(f"Error context: {processed_error.context}")
