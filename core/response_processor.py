import asyncio
import json
import uuid
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Literal,
    Optional,
    Union,
    cast,
)

from claude_agent_sdk import SystemMessage
from claude_agent_sdk.types import (
    AssistantMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from loguru import logger

from configs import app_config
from core.error_processor import ErrorProcessor
from core.tool import ToolResult
from core.utils.json_helpers import (
    format_for_yield,
    safe_json_parse,
    to_json_string,
)
from models.thread import Threads

# XML结果添加策略的类型别名
XmlAddingStrategy = Literal["user_message", "assistant_message", "inline_edit"]


@dataclass
class ToolExecutionContext:
    """工具执行上下文，包含调用详情、结果和显示信息。"""

    tool_call: dict[str, Any]
    tool_index: int
    result: Optional[ToolResult] = None
    function_name: Optional[str] = None
    error: Optional[Exception] = None
    assistant_message_id: Optional[str] = None


@dataclass
class ProcessorConfig:
    """
    响应处理和工具执行的配置。

    该类控制LLM响应的处理方式，包括工具调用的检测、执行和结果处理。

    属性:
        execute_on_stream: 对于流式响应，是即时执行工具还是在结束时执行
    """

    execute_on_stream: bool = False


class ResponseProcessor:
    """处理LLM响应，提取并执行工具调用。"""

    def __init__(
        self,
        add_message_callback: Callable,
        agent_config: Optional[dict] = None,
    ):
        """初始化ResponseProcessor。

        参数:
            add_message_callback: 向线程添加消息的回调函数。
                必须返回完整保存的消息对象(dict)或None。
            agent_config: 可选的代理配置，包含版本信息
        """
        self.add_message = add_message_callback

        self.agent_config = agent_config

    async def _yield_message(self, message_obj: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """辅助方法：以适当的格式生成消息。

        确保内容和元数据为JSON字符串，以保证客户端兼容性。
        """
        if message_obj:
            return format_for_yield(message_obj)
        return None

    def _serialize_model_response(self, model_response) -> dict[str, Any]:
        """将ModelResponse对象转换为可JSON序列化的字典。

        参数:
            model_response: ModelResponse对象

        返回:
            ModelResponse的字典表示
        """
        try:
            # 尝试使用model_dump方法（Pydantic v2）
            if hasattr(model_response, "model_dump"):
                return model_response.model_dump()

            # 尝试使用dict方法（Pydantic v1）
            elif hasattr(model_response, "dict"):
                return model_response.dict()

            # 回退：手动提取常见属性
            else:
                result = {}

                # 常见的LiteLLM ModelResponse属性
                for attr in [
                    "id",
                    "object",
                    "created",
                    "model",
                    "choices",
                    "usage",
                    "system_fingerprint",
                ]:
                    if hasattr(model_response, attr):
                        value = getattr(model_response, attr)
                        # 递归处理嵌套对象
                        if hasattr(value, "model_dump"):
                            result[attr] = value.model_dump()
                        elif hasattr(value, "dict"):
                            result[attr] = value.dict()
                        elif isinstance(value, list):
                            result[attr] = [
                                item.model_dump()
                                if hasattr(item, "model_dump")
                                else item.dict()
                                if hasattr(item, "dict")
                                else item
                                for item in value
                            ]
                        else:
                            result[attr] = value

                return result

        except Exception as e:
            logger.warning(f"序列化ModelResponse失败: {str(e)}, 回退到字符串表示")
            # 最终回退：转换为字符串
            return {"raw_response": str(model_response), "serialization_error": str(e)}

    async def _add_message_with_agent_info(
        self,
        thread_id: str,
        type: str,
        content: dict[str, Any] | list[Any] | str,
        is_llm_message: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """辅助方法：在可用时添加包含代理版本信息的消息。"""
        agent_id = None
        agent_version_id = None

        if self.agent_config:
            agent_id = self.agent_config.get("agent_id")
            agent_version_id = self.agent_config.get("current_version_id")

        return await self.add_message(
            thread_id=thread_id,
            type=type,
            content=content,
            is_llm_message=is_llm_message,
            metadata=metadata,
            agent_id=agent_id,
            agent_version_id=agent_version_id,
        )

    async def process_streaming_response(
        self,
        llm_response: AsyncGenerator,
        thread_id: str,
        prompt_messages: list[dict[str, Any]],
        llm_model: str,
        config: ProcessorConfig = ProcessorConfig(),
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """处理流式LLM响应，处理工具调用和执行。

        参数:
            llm_response: LLM的流式响应
            thread_id: 对话线程的ID
            prompt_messages: 发送给LLM的消息列表（提示）
            llm_model: 使用的LLM模型名称
            config: 解析和执行的配置
            cancellation_event: 取消事件

        生成:
            完整的消息对象，匹配数据库模式，除了内容块。
        """
        logger.info(f"开始处理Claude Code流式响应 - thread: {thread_id}")

        # 初始化取消事件
        if cancellation_event is None:
            cancellation_event = asyncio.Event()

        # 初始化状态变量
        accumulated_content = ""  # 累积的文本内容
        content_blocks = {}  # index -> block_data (text或tool_use)
        current_message_id = None  # 当前assistant消息ID
        usage_data = {}  # Token使用统计
        finish_reason = None  # 完成原因
        last_assistant_message_object = None  # 最后保存的assistant消息对象
        turn_count = 0  # 对话轮次计数
        session_id = None  # claude code 会话ID

        # 存储完整的响应对象用于billing
        final_llm_response = None
        first_chunk_time = None
        last_chunk_time = None
        llm_response_end_saved = False

        # 生成运行ID
        thread_run_id = str(uuid.uuid4())
        llm_response_id = str(uuid.uuid4())

        logger.info(f"运行ID: thread_run_id={thread_run_id}, llm_response_id={llm_response_id}")

        try:
            # --- 保存并yield启动事件 ---
            start_content = {
                "status_type": "thread_run_start",
                "thread_run_id": thread_run_id,
            }
            start_msg_obj = await self.add_message(
                thread_id=thread_id,
                type="status",
                content=start_content,
                is_llm_message=False,
                metadata={"thread_run_id": thread_run_id},
            )
            if start_msg_obj:
                yield format_for_yield(start_msg_obj)

            llm_start_content = {
                "llm_response_id": llm_response_id,
                "model": llm_model,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            llm_start_msg_obj = await self.add_message(
                thread_id=thread_id,
                type="llm_response_start",
                content=llm_start_content,
                is_llm_message=False,
                metadata={
                    "thread_run_id": thread_run_id,
                    "llm_response_id": llm_response_id,
                },
            )
            if llm_start_msg_obj:
                yield format_for_yield(llm_start_msg_obj)
                logger.info("已保存llm_response_start")
            # --- 启动事件结束 ---

            __sequence = 0  # 消息序列号

            # 设置debug文件保存（如果启用）
            debug_file = None
            debug_file_json = None
            raw_chunks_data = []  # 存储所有chunk数据用于JSONL导出

            if app_config.DEBUG:
                debug_dir = Path("debug_streams")
                debug_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                debug_file = debug_dir / f"stream_{thread_id[:8]}_{timestamp}.txt"
                debug_file_json = debug_dir / f"stream_{thread_id[:8]}_{timestamp}.jsonl"

                logger.info(f"保存原始流输出到: {debug_file}")

            chunk_count = 0
            tool_index = 0  # 工具调用索引

            # --- 主循环：处理Claude Code流式响应 ---
            async for chunk in llm_response:
                # 检查取消信号
                if cancellation_event.is_set():
                    logger.info(f"收到取消信号，停止处理 - thread: {thread_id}")
                    finish_reason = "cancelled"
                    break

                chunk_count += 1

                # 跟踪时间
                current_time = datetime.now().timestamp()
                if first_chunk_time is None:
                    first_chunk_time = current_time
                last_chunk_time = current_time

                # 获取chunk类型
                chunk_type = type(chunk).__name__

                # 定期记录日志
                if chunk_count == 1 or (chunk_count % 100 == 0):
                    logger.debug(f"处理chunk #{chunk_count}, type={chunk_type}")

                # 保存原始chunk数据用于调试（如果启用）
                if app_config.DEBUG:
                    try:
                        chunk_data = {
                            "chunk_num": chunk_count,
                            "timestamp": current_time,
                            "chunk_type": chunk_type,
                            "chunk_str": str(chunk)[:200],  # 前200字符
                        }
                        raw_chunks_data.append(chunk_data)

                        # 增量写入JSONL文件
                        with open(debug_file_json, "a", encoding="utf-8") as f:
                            f.write(json.dumps(chunk_data, ensure_ascii=False) + "\n")
                    except Exception as e:
                        logger.debug(f"保存chunk数据错误: {e}")

                # --- 1. 处理 SystemMessage（初始化信息） ---
                if chunk_type == "SystemMessage":
                    logger.debug("收到系统初始化消息")
                    system_message = cast(SystemMessage, chunk)
                    session_id = system_message.data.get("session_id")
                    if session_id:
                        Threads.update_session_id(thread_id, session_id)
                    continue

                # --- 2. 处理 StreamEvent（流式事件） ---
                elif chunk_type == "StreamEvent":
                    event = chunk.event
                    event_type = event.get("type")

                    # 2.1 message_start - 记录message_id和usage
                    if event_type == "message_start":
                        turn_count += 1
                        current_message_id = event["message"]["id"]
                        if "usage" in event["message"]:
                            usage_data = event["message"]["usage"]
                        logger.info(f"开始第{turn_count}轮消息: {current_message_id}")

                    # 2.2 content_block_start - 文本块或工具调用块开始
                    elif event_type == "content_block_start":
                        index = event["index"]
                        content_block = event["content_block"]
                        block_type = content_block["type"]

                        if block_type == "text":
                            # 文本块开始
                            content_blocks[index] = {"type": "text", "text": ""}
                            logger.debug(f"文本块开始 (index={index})")

                        elif block_type == "tool_use":
                            # 工具调用块开始
                            tool_call_id = content_block["id"]
                            tool_name = content_block["name"]
                            content_blocks[index] = {
                                "type": "tool_use",
                                "id": tool_call_id,
                                "name": tool_name,
                                "input": "",  # 将累积JSON片段
                            }
                            logger.info(f"工具调用开始: {tool_name} (id={tool_call_id})")

                            # yield tool_started 状态消息
                            tool_started_content = {
                                "status_type": "tool_started",
                                "tool_call_id": tool_call_id,
                                "function_name": tool_name,
                                "tool_index": tool_index,
                            }
                            tool_started_msg = await self.add_message(
                                thread_id=thread_id,
                                type="status",
                                content=tool_started_content,
                                is_llm_message=False,
                                metadata={"thread_run_id": thread_run_id},
                            )
                            if tool_started_msg:
                                yield format_for_yield(tool_started_msg)

                            tool_index += 1

                    # 2.3 content_block_delta - 内容增量
                    elif event_type == "content_block_delta":
                        index = event["index"]
                        delta = event["delta"]
                        delta_type = delta["type"]

                        if delta_type == "text_delta":
                            # 文本增量
                            text_chunk = delta["text"]
                            accumulated_content += text_chunk
                            if index in content_blocks and content_blocks[index]["type"] == "text":
                                content_blocks[index]["text"] += text_chunk

                            # yield文本内容
                            now_chunk = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            yield {
                                "sequence": __sequence,
                                "message_id": None,
                                "thread_id": thread_id,
                                "type": "assistant",
                                "is_llm_message": True,
                                "content": to_json_string({"role": "assistant", "content": text_chunk}),
                                "metadata": to_json_string(
                                    {
                                        "stream_status": "chunk",
                                        "thread_run_id": thread_run_id,
                                    }
                                ),
                                "created_at": now_chunk,
                                "updated_at": now_chunk,
                            }
                            __sequence += 1

                        elif delta_type == "input_json_delta":
                            # 工具参数JSON增量
                            partial_json = delta["partial_json"]
                            if index in content_blocks and content_blocks[index]["type"] == "tool_use":
                                content_blocks[index]["input"] += partial_json

                    # 2.4 content_block_stop - 内容块结束
                    elif event_type == "content_block_stop":
                        index = event["index"]
                        block = content_blocks.get(index)

                        if block and block["type"] == "tool_use":
                            # 工具调用完整，解析参数
                            try:
                                tool_input = json.loads(block["input"])
                                block["parsed_input"] = tool_input
                                logger.debug(f"工具参数解析完成: {block['name']}")
                            except json.JSONDecodeError as e:
                                logger.error(f"工具参数JSON解析失败: {e}")
                                block["parsed_input"] = {}

                    # 2.5 message_delta - 消息增量（usage和stop_reason）
                    elif event_type == "message_delta":
                        delta = event.get("delta", {})
                        if "stop_reason" in delta:
                            finish_reason = delta["stop_reason"]
                            logger.debug(f"finish_reason={finish_reason}")
                        if "usage" in event:
                            usage_data.update(event["usage"])

                    # 2.6 message_stop - 消息结束
                    elif event_type == "message_stop":
                        logger.debug(f"消息流结束 (第{turn_count}轮)")

                # --- 3. 处理 AssistantMessage（完整消息） ---
                elif chunk_type == "AssistantMessage":
                    # 保存assistant消息到DB
                    content_data = chunk.content
                    message_content = self._format_assistant_message_content(content_data)

                    last_assistant_message_object = await self._add_message_with_agent_info(
                        thread_id=thread_id,
                        type="assistant",
                        content=message_content,
                        is_llm_message=True,
                        metadata={"thread_run_id": thread_run_id},
                    )

                    if last_assistant_message_object:
                        # yield完整消息
                        yield_metadata = last_assistant_message_object.get("metadata", {})
                        yield_metadata["stream_status"] = "complete"
                        yield_message = last_assistant_message_object.copy()
                        yield_message["metadata"] = yield_metadata
                        yield format_for_yield(yield_message)

                        logger.info(f"已保存assistant消息: {last_assistant_message_object.get('message_id')}")

                # --- 4. 处理 UserMessage（工具执行结果） ---
                elif chunk_type == "UserMessage":
                    # 提取工具结果（content是ToolResultBlock列表）
                    for block in chunk.content:
                        if hasattr(block, "tool_use_id"):
                            tool_result_content = {
                                "tool_use_id": block.tool_use_id,
                                "content": block.content,
                                "is_error": getattr(block, "is_error", None),
                            }

                            # 保存tool result到DB
                            tool_result_msg = await self._add_message_with_agent_info(
                                thread_id=thread_id,
                                type="tool_result",
                                content=tool_result_content,
                                is_llm_message=False,
                                metadata={"thread_run_id": thread_run_id},
                            )

                            # yield tool_completed状态
                            if tool_result_msg:
                                # yield工具完成状态
                                tool_completed_content = {
                                    "status_type": "tool_completed",
                                    "tool_call_id": block.tool_use_id,
                                    "success": not getattr(block, "is_error", False),
                                }
                                tool_completed_msg = await self.add_message(
                                    thread_id=thread_id,
                                    type="status",
                                    content=tool_completed_content,
                                    is_llm_message=False,
                                    metadata={"thread_run_id": thread_run_id},
                                )
                                if tool_completed_msg:
                                    yield format_for_yield(tool_completed_msg)

                                # yield工具结果消息
                                yield format_for_yield(tool_result_msg)

                            logger.info(f"已保存工具结果: {block.tool_use_id}")

                # --- 5. 处理 ResultMessage（最终结果） ---
                elif chunk_type == "ResultMessage":
                    # 提取统计信息
                    final_usage = chunk.usage if hasattr(chunk, "usage") else usage_data
                    total_cost = getattr(chunk, "total_cost_usd", 0)
                    num_turns = getattr(chunk, "num_turns", turn_count)

                    logger.info(f"对话完成: {num_turns}轮, 成本=${total_cost:.5f}")
                    logger.info(f"Token使用: {final_usage}")

                    # 保存到final_llm_response以便后续保存llm_response_end
                    final_llm_response = chunk

            # --- 流处理结束 ---
            logger.info(f"流处理完成. 总chunks: {chunk_count}, finish_reason: {finish_reason}")
            logger.info(f"累积内容长度: {len(accumulated_content)} 字符, 对话轮数: {turn_count}")

            # 保存debug摘要（如果启用）
            if app_config.DEBUG:
                try:
                    summary = {
                        "thread_id": thread_id,
                        "thread_run_id": thread_run_id,
                        "total_chunks": chunk_count,
                        "turn_count": turn_count,
                        "finish_reason": finish_reason,
                        "accumulated_content_length": len(accumulated_content),
                        "tool_calls_count": len([b for b in content_blocks.values() if b.get("type") == "tool_use"]),
                        "first_chunk_time": first_chunk_time,
                        "last_chunk_time": last_chunk_time,
                        "final_usage": usage_data,
                    }

                    # 计算响应时间
                    if first_chunk_time and last_chunk_time:
                        summary["response_time_ms"] = (last_chunk_time - first_chunk_time) * 1000

                    # 写入摘要到文本文件
                    with open(debug_file, "w", encoding="utf-8") as f:
                        f.write("=" * 80 + "\n")
                        f.write("CLAUDE CODE STREAM DEBUG SUMMARY\n")
                        f.write("=" * 80 + "\n\n")
                        f.write(json.dumps(summary, indent=2, ensure_ascii=False) + "\n\n")
                        f.write("=" * 80 + "\n")
                        f.write("ACCUMULATED CONTENT\n")
                        f.write("=" * 80 + "\n\n")
                        f.write(accumulated_content + "\n\n")
                        f.write("=" * 80 + "\n")
                        f.write(f"Total chunks: {chunk_count}\n")
                        f.write(f"Content blocks: {len(content_blocks)}\n")

                    logger.info(f"已保存stream debug文件: {debug_file} 和 {debug_file_json}")
                except Exception as e:
                    logger.warning(f"保存stream debug摘要错误: {e}")

            # 如果有时间数据，计算响应时间
            response_ms = None
            if first_chunk_time and last_chunk_time:
                response_ms = (last_chunk_time - first_chunk_time) * 1000

            # 验证usage已捕获
            if not usage_data:
                logger.warning("未从流中捕获usage数据")

            # --- yield finish状态 ---
            if finish_reason:
                finish_content = {
                    "status_type": "finish",
                    "finish_reason": finish_reason,
                }
                finish_msg_obj = await self.add_message(
                    thread_id=thread_id,
                    type="status",
                    content=finish_content,
                    is_llm_message=False,
                    metadata={"thread_run_id": thread_run_id},
                )
                if finish_msg_obj:
                    yield format_for_yield(finish_msg_obj)
                logger.info(f"yield finish状态: {finish_reason}")

            # --- 保存并yield llm_response_end ---
            if last_assistant_message_object:
                try:
                    # 构建llm_response_end内容
                    logger.info("构建Claude Code llm_response_end")
                    llm_end_content = self._serialize_claude_code_response(final_llm_response, usage_data)

                    # 添加streaming标志和响应时间
                    llm_end_content["streaming"] = True
                    if response_ms:
                        llm_end_content["response_ms"] = response_ms
                    llm_end_content["llm_response_id"] = llm_response_id

                    # 保存llm_response_end消息
                    llm_end_msg_obj = await self.add_message(
                        thread_id=thread_id,
                        type="llm_response_end",
                        content=llm_end_content,
                        is_llm_message=False,
                        metadata={
                            "thread_run_id": thread_run_id,
                            "llm_response_id": llm_response_id,
                        },
                    )
                    llm_response_end_saved = True
                    # Yield到stream用于实时更新
                    if llm_end_msg_obj:
                        yield format_for_yield(llm_end_msg_obj)
                    logger.info("llm_response_end已保存")
                except Exception as e:
                    logger.error(f"保存llm_response_end错误: {str(e)}")

        except Exception as e:
            # 使用ErrorProcessor进行一致的错误处理
            processed_error = ErrorProcessor.process_system_error(e, context={"thread_id": thread_id})
            ErrorProcessor.log_error(processed_error)

            # 保存并生成错误状态消息
            err_content = {
                "role": "system",
                "status_type": "error",
                "message": processed_error.message,
            }
            err_msg_obj = await self.add_message(
                thread_id=thread_id,
                type="status",
                content=err_content,
                is_llm_message=False,
                metadata={"thread_run_id": thread_run_id if "thread_run_id" in locals() else None},
            )
            if err_msg_obj:
                yield format_for_yield(err_msg_obj)
            raise

        finally:
            # IMPORTANT: Finally块即使在stream停止时也会运行（GeneratorExit）
            # 不能在这里yield - 只能静默地保存到DB用于billing/usage跟踪

            # 阶段3：资源清理 - 取消pending任务并关闭generator
            try:
                # 尝试关闭LLM响应generator（如果支持aclose()）
                # 这有助于停止底层的HTTP连接
                if hasattr(llm_response, "aclose"):
                    try:
                        await llm_response.aclose()
                        logger.debug(f"已关闭LLM响应generator - thread: {thread_id}")
                    except Exception as close_err:
                        logger.debug(f"关闭LLM响应generator错误（可能不支持aclose）: {close_err}")
                elif hasattr(llm_response, "close"):
                    try:
                        llm_response.close()
                        logger.debug(f"已关闭LLM响应generator (sync close) - thread: {thread_id}")
                    except Exception as close_err:
                        logger.debug(f"关闭LLM响应generator错误 (sync): {close_err}")
            except Exception as cleanup_err:
                logger.warning(f"资源清理错误: {cleanup_err}")

            # Billing保护：如果llm_response_end还没保存，在finally块中保存
            if not llm_response_end_saved and last_assistant_message_object:
                try:
                    logger.info("BULLETPROOF BILLING: 在finally块中保存llm_response_end")
                    if final_llm_response and usage_data:
                        logger.info("使用LLM响应中的精确usage")
                        llm_end_content = self._serialize_claude_code_response(final_llm_response, usage_data)
                    else:
                        logger.warning("没有LLM响应使用量 - 为计费估算token使用量")
                        llm_end_content = {"model": llm_model, "usage": {}}

                    llm_end_content["streaming"] = True
                    llm_end_content["llm_response_id"] = llm_response_id

                    response_ms = None
                    if first_chunk_time and last_chunk_time:
                        response_ms = int((last_chunk_time - first_chunk_time) * 1000)
                        llm_end_content["response_ms"] = response_ms

                    usage_info = llm_end_content.get("usage", {})
                    is_estimated = usage_info.get("estimated", False)

                    # 保存（不yield）
                    llm_end_msg_obj = await self.add_message(
                        thread_id=thread_id,
                        type="llm_response_end",
                        content=llm_end_content,
                        is_llm_message=False,
                        metadata={
                            "thread_run_id": thread_run_id,
                            "llm_response_id": llm_response_id,
                        },
                    )
                    logger.info("llm_response_end已在finally块中保存")
                    llm_response_end_saved = True
                except Exception as finally_err:
                    logger.error(f"在finally块中保存llm_response_end失败: {str(finally_err)}")

            # Phase 4: 保存并yield thread_run_end状态
            # 注意：只在auto_continue_count == 0时保存thread_run_end（即最外层调用）
            # Claude Code中不使用auto_continue，所以始终保存
            try:
                end_content = {"status_type": "thread_run_end"}
                end_msg_obj = await self.add_message(
                    thread_id=thread_id,
                    type="status",
                    content=end_content,
                    is_llm_message=False,
                    metadata={"thread_run_id": thread_run_id if "thread_run_id" in locals() else None},
                )
                # 不要yield - finally块中的yield会导致问题
                logger.info("thread_run_end已保存")
            except Exception as end_err:
                logger.error(f"保存thread_run_end错误: {str(end_err)}")
                # 不要re-raise - 让主异常传播

    async def process_non_streaming_response(
        self,
        llm_response: Any,
        thread_id: str,
        prompt_messages: list[dict[str, Any]],
        llm_model: str,
        config: ProcessorConfig = ProcessorConfig(),
    ) -> AsyncGenerator[dict[str, Any], None]:
        """处理非流式LLM响应，处理工具调用和执行。

        参数:
            llm_response: LLM的响应
            thread_id: 对话线程的ID
            prompt_messages: 发送给LLM的消息列表（提示）
            llm_model: 使用的LLM模型名称
            config: 解析和执行的配置

        生成:
            匹配数据库模式的完整消息对象。
        """
        content = ""
        thread_run_id = str(uuid.uuid4())
        assistant_message_object = None
        finish_reason = None
        native_tool_calls_for_message = []

        try:
            # 保存并生成thread_run_start状态消息
            start_content = {
                "status_type": "thread_run_start",
                "thread_run_id": thread_run_id,
            }
            start_msg_obj = await self.add_message(
                thread_id=thread_id,
                type="status",
                content=start_content,
                is_llm_message=False,
                metadata={"thread_run_id": thread_run_id},
            )
            if start_msg_obj:
                yield format_for_yield(start_msg_obj)

            # 提取finish_reason、内容、工具调用
            if hasattr(llm_response, "choices") and llm_response.choices:
                if hasattr(llm_response.choices[0], "finish_reason"):
                    finish_reason = llm_response.choices[0].finish_reason
                    logger.debug(f"非流式finish_reason: {finish_reason}")
                response_message = (
                    llm_response.choices[0].message if hasattr(llm_response.choices[0], "message") else None
                )
                if response_message:
                    if hasattr(response_message, "content") and response_message.content:
                        content = response_message.content

            # --- 保存并生成最终助手消息 ---
            message_data = {
                "role": "assistant",
                "content": content,
                "tool_calls": native_tool_calls_for_message or None,
            }
            assistant_message_object = await self._add_message_with_agent_info(
                thread_id=thread_id,
                type="assistant",
                content=message_data,
                is_llm_message=True,
                metadata={"thread_run_id": thread_run_id},
            )
            if assistant_message_object:
                yield assistant_message_object
            else:
                logger.error(f"为线程 {thread_id} 保存非流式助手消息失败")
                err_content = {
                    "role": "system",
                    "status_type": "error",
                    "message": "保存助手消息失败",
                }
                err_msg_obj = await self.add_message(
                    thread_id=thread_id,
                    type="status",
                    content=err_content,
                    is_llm_message=False,
                    metadata={"thread_run_id": thread_run_id},
                )
                if err_msg_obj:
                    yield format_for_yield(err_msg_obj)

            # --- 保存并生成最终状态 ---
            if finish_reason:
                finish_content = {
                    "status_type": "finish",
                    "finish_reason": finish_reason,
                }
                finish_msg_obj = await self.add_message(
                    thread_id=thread_id,
                    type="status",
                    content=finish_content,
                    is_llm_message=False,
                    metadata={"thread_run_id": thread_run_id},
                )
                if finish_msg_obj:
                    yield format_for_yield(finish_msg_obj)

            # --- 保存并生成assistant_response_end ---
            if assistant_message_object:  # 仅在保存了助手消息时保存
                try:
                    # 将LiteLLM ModelResponse转换为可JSON序列化的字典
                    response_dict = self._serialize_model_response(llm_response)

                    # 在内容中保存序列化的响应对象
                    await self.add_message(
                        thread_id=thread_id,
                        type="assistant_response_end",
                        content=response_dict,
                        is_llm_message=False,
                        metadata={"thread_run_id": thread_run_id},
                    )
                    logger.debug("非流式响应的助手响应结束已保存")
                except Exception as e:
                    logger.error(f"为非流式保存助手响应结束时出错: {str(e)}")

        except Exception as e:
            # 使用ErrorProcessor进行一致的错误处理
            processed_error = ErrorProcessor.process_system_error(e, context={"thread_id": thread_id})
            ErrorProcessor.log_error(processed_error)

            # 保存并生成错误状态
            err_content = {
                "role": "system",
                "status_type": "error",
                "message": processed_error.message,
            }
            err_msg_obj = await self.add_message(
                thread_id=thread_id,
                type="status",
                content=err_content,
                is_llm_message=False,
                metadata={"thread_run_id": thread_run_id if "thread_run_id" in locals() else None},
            )
            if err_msg_obj:
                yield format_for_yield(err_msg_obj)

            raise

        finally:
            # 保存并生成最终的thread_run_end状态
            usage = llm_response.usage if hasattr(llm_response, "usage") else None

            end_content = {"status_type": "thread_run_end"}

            end_msg_obj = await self.add_message(
                thread_id=thread_id,
                type="status",
                content=end_content,
                is_llm_message=False,
                metadata={"thread_run_id": thread_run_id if "thread_run_id" in locals() else None},
            )
            if end_msg_obj:
                yield format_for_yield(end_msg_obj)

    async def _add_tool_result(
        self,
        thread_id: str,
        tool_call: dict[str, Any],
        result: ToolResult,
        strategy: Union[XmlAddingStrategy, str] = "assistant_message",
        assistant_message_id: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:  # 返回完整的消息对象
        """根据指定格式将工具结果添加到对话线程。

        该方法格式化工具结果并将其添加到对话历史记录中，
        使其在后续交互中对LLM可见。结果可以作为
        原生工具消息（OpenAI格式）或带有指定角色（用户或助手）的XML包装内容添加。

        参数:
            thread_id: 对话线程的ID
            tool_call: 产生此结果的原始工具调用
            result: 工具执行的结果
            strategy: 如何将XML工具结果添加到对话
                     ("user_message", "assistant_message", 或 "inline_edit")
            assistant_message_id: 生成此工具调用的助手消息ID
        """
        try:
            message_obj = None  # 初始化message_obj

            # 如果提供了assistant_message_id，则创建包含它的元数据
            metadata = {}
            if assistant_message_id:
                metadata["assistant_message_id"] = assistant_message_id
                logger.debug(f"将工具结果链接到助手消息: {assistant_message_id}")

            # 检查这是否是原生函数调用（具有id字段）
            if "id" in tool_call:
                # 根据OpenAI规范格式化为适当的工具消息
                function_name = tool_call.get("function_name", "")

                # 格式化工具结果内容 - 工具角色需要字符串内容
                if isinstance(result, str):
                    content = result
                elif hasattr(result, "output"):
                    # 如果是ToolResult对象
                    if isinstance(result.output, (dict, list)):
                        # 如果输出已经是dict或list，转换为JSON字符串
                        content = json.dumps(result.output)
                    else:
                        # 否则仅使用字符串表示
                        content = str(result.output)
                else:
                    # 回退到整个结果的字符串表示
                    content = str(result)

                logger.debug(f"格式化的工具结果内容: {content[:100]}...")

                # 创建具有适当格式的工具响应消息
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": function_name,
                    "content": content,
                }

                logger.debug(f"为tool_call_id={tool_call['id']}添加原生工具结果，角色为tool")

                # 作为工具消息添加到对话历史记录
                # 这使结果在下一轮对LLM可见
                message_obj = await self.add_message(
                    thread_id=thread_id,
                    type="tool",  # 工具响应的特殊类型
                    content=tool_message,
                    is_llm_message=True,
                    metadata=metadata,
                )
                return message_obj  # 返回完整的消息对象

            # 对于XML和其他非原生工具，使用新的结构化格式
            # 根据策略确定消息角色
            result_role = "user" if strategy == "user_message" else "assistant"

            # 创建两个版本的结构化结果
            # 1. 用于前端的富版本
            structured_result_for_frontend = self._create_structured_tool_result(
                tool_call, result, parsing_details=None, for_llm=False
            )
            # 2. 用于LLM的简洁版本
            structured_result_for_llm = self._create_structured_tool_result(
                tool_call, result, parsing_details=None, for_llm=True
            )

            # 将具有适当角色的消息添加到对话历史记录
            # 这允许LLM在后续交互中看到工具结果
            result_message_for_llm = {
                "role": result_role,
                "content": json.dumps(structured_result_for_llm),
            }

            # 将富内容添加到元数据以供前端使用
            if metadata is None:
                metadata = {}
            metadata["frontend_content"] = structured_result_for_frontend

            message_obj = await self._add_message_with_agent_info(
                thread_id=thread_id,
                type="tool",
                content=result_message_for_llm,  # 保存LLM友好版本
                is_llm_message=True,
                metadata=metadata,
            )

            # 如果消息已保存，在返回前在内存中为前端修改它
            if message_obj:
                # 前端期望在'content'字段中有富内容。
                # 数据库在metadata.frontend_content中有富内容。
                # 让我们重构消息以供生成。
                message_for_yield = message_obj.copy()
                message_for_yield["content"] = structured_result_for_frontend
                return message_for_yield

            return message_obj  # 返回修改后的消息对象
        except Exception as e:
            logger.error(f"添加工具结果时出错: {str(e)}", exc_info=True)
            # 回退到简单消息
            try:
                fallback_message = {"role": "user", "content": str(result)}
                message_obj = await self.add_message(
                    thread_id=thread_id,
                    type="tool",
                    content=fallback_message,
                    is_llm_message=True,
                    metadata={"assistant_message_id": assistant_message_id} if assistant_message_id else {},
                )
                return message_obj  # 返回完整的消息对象
            except Exception as e2:
                logger.error(f"即使使用回退消息也失败: {str(e2)}", exc_info=True)
                return None  # 出错时返回None

    def _create_structured_tool_result(
        self,
        tool_call: dict[str, Any],
        result: ToolResult,
        parsing_details: Optional[dict[str, Any]] = None,
        for_llm: bool = False,
    ):
        """创建与工具无关且提供丰富信息的结构化工具结果格式。

        参数:
            tool_call: 被执行的原始工具调用
            result: 工具执行的结果
            parsing_details: XML调用的可选解析详情
            for_llm: 如果为True，为LLM上下文创建简洁版本。

        返回:
            包含工具执行信息的结构化字典
        """
        # 提取工具信息
        function_name = tool_call.get("function_name", "unknown")
        arguments = tool_call.get("arguments", {})
        tool_call_id = tool_call.get("id")

        # 处理输出 - 如果是JSON字符串，则解析回对象
        output = result.output if hasattr(result, "output") else str(result)
        if isinstance(output, str):
            try:
                # 尝试解析为JSON以向前端提供结构化数据
                parsed_output = safe_json_parse(output)
                # 如果解析成功且得到dict/list，则使用解析版本
                if isinstance(parsed_output, (dict, list)):
                    output = parsed_output
                # 否则保留原始字符串
            except Exception:
                # 如果解析失败，保留原始字符串
                pass

        structured_result_v1 = {
            "tool_execution": {
                "function_name": function_name,
                "tool_call_id": tool_call_id,
                "arguments": arguments,
                "result": {
                    "success": result.success if hasattr(result, "success") else True,
                    "output": output,
                    "error": getattr(result, "error", None) if hasattr(result, "error") else None,
                },
            }
        }

        return structured_result_v1

    def _create_tool_context(
        self,
        tool_call: dict[str, Any],
        tool_index: int,
        assistant_message_id: Optional[str] = None,
    ) -> ToolExecutionContext:
        """创建包含显示名称和解析详情的工具执行上下文。"""
        context = ToolExecutionContext(
            tool_call=tool_call,
            tool_index=tool_index,
            assistant_message_id=assistant_message_id,
        )

        context.function_name = tool_call.get("function_name", "unknown")

        return context

    async def _yield_and_save_tool_started(
        self, context: ToolExecutionContext, thread_id: str, thread_run_id: str
    ) -> Optional[dict[str, Any]]:
        """格式化、保存并返回工具开始状态消息。"""
        tool_name = context.function_name
        content = {
            "role": "assistant",
            "status_type": "tool_started",
            "function_name": context.function_name,
            "message": f"开始执行 {tool_name}",
            "tool_index": context.tool_index,
            "tool_call_id": context.tool_call.get("id"),  # 如果是原生的，包含tool_call ID
        }
        metadata = {"thread_run_id": thread_run_id}
        saved_message_obj = await self.add_message(
            thread_id=thread_id,
            type="status",
            content=content,
            is_llm_message=False,
            metadata=metadata,
        )
        return saved_message_obj  # 返回完整对象（如果保存失败则返回None）

    async def _yield_and_save_tool_completed(
        self,
        context: ToolExecutionContext,
        tool_message_id: Optional[str],
        thread_id: str,
        thread_run_id: str,
    ) -> Optional[dict[str, Any]]:
        """格式化、保存并返回工具完成/失败状态消息。"""
        if not context.result:
            # 如果结果缺失（例如执行失败），委托给错误保存
            return await self._yield_and_save_tool_error(context, thread_id, thread_run_id)

        tool_name = context.function_name
        status_type = "tool_completed" if context.result.success else "tool_failed"
        message_text = f"工具 {tool_name} {'成功完成' if context.result.success else '失败'}"

        content = {
            "role": "assistant",
            "status_type": status_type,
            "function_name": context.function_name,
            "message": message_text,
            "tool_index": context.tool_index,
            "tool_call_id": context.tool_call.get("id"),
        }
        metadata = {"thread_run_id": thread_run_id}
        # 如果可用且成功，将*实际*工具结果消息ID添加到元数据
        if context.result.success and tool_message_id:
            metadata["linked_tool_result_message_id"] = tool_message_id

        # <<< 添加：如果这是终止工具，则发出信号 >>>
        if context.function_name in ["ask", "complete"]:
            metadata["agent_should_terminate"] = "true"
            logger.debug(f"使用终止信号标记工具状态 '{context.function_name}'。")
        # <<< 结束添加 >>>

        saved_message_obj = await self.add_message(
            thread_id=thread_id,
            type="status",
            content=content,
            is_llm_message=False,
            metadata=metadata,
        )
        return saved_message_obj

    async def _yield_and_save_tool_error(
        self, context: ToolExecutionContext, thread_id: str, thread_run_id: str
    ) -> Optional[dict[str, Any]]:
        """格式化、保存并返回工具错误状态消息。"""
        error_msg = str(context.error) if context.error else "工具执行期间未知错误"
        tool_name = context.function_name
        content = {
            "role": "assistant",
            "status_type": "tool_error",
            "function_name": context.function_name,
            "message": f"执行工具 {tool_name} 时出错: {error_msg}",
            "tool_index": context.tool_index,
            "tool_call_id": context.tool_call.get("id"),
        }
        metadata = {"thread_run_id": thread_run_id}
        # 使用is_llm_message=False保存状态消息
        saved_message_obj = await self.add_message(
            thread_id=thread_id,
            type="status",
            content=content,
            is_llm_message=False,
            metadata=metadata,
        )
        return saved_message_obj

    def _format_assistant_message_content(self, content_blocks: list) -> dict:
        """格式化assistant消息内容，兼容TextBlock和ToolUseBlock。

        Args:
            content_blocks: Claude Code返回的content blocks列表

        Returns:
            格式化后的消息内容字典
        """
        text_parts = []
        tool_calls = []

        for block in content_blocks:
            block_type = type(block).__name__
            if block_type == "TextBlock":
                text_parts.append(block.text)
            elif block_type == "ToolUseBlock":
                tool_calls.append({"id": block.id, "name": block.name, "input": block.input})

        result = {"role": "assistant", "content": "".join(text_parts)}

        if tool_calls:
            result["tool_calls"] = tool_calls

        return result

    def _serialize_claude_code_response(self, claude_response, usage_data: dict) -> dict:
        """序列化Claude Code响应对象用于保存llm_response_end。

        Args:
            claude_response: Claude Code的ResultMessage或其他响应对象
            usage_data: 收集的usage数据

        Returns:
            序列化后的响应内容
        """
        result = {
            "model": getattr(claude_response, "model", "kimi-for-coding"),
            "usage": usage_data or {},
        }

        # 如果是ResultMessage，提取更多信息
        if hasattr(claude_response, "total_cost_usd"):
            result["total_cost_usd"] = claude_response.total_cost_usd
        if hasattr(claude_response, "num_turns"):
            result["num_turns"] = claude_response.num_turns

        return result
