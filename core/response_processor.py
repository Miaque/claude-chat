import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)
from zoneinfo import ZoneInfo

from loguru import logger

from core.error_processor import ErrorProcessor
from core.tool import ToolResult
from core.utils.json_helpers import (
    ensure_dict,
    format_for_yield,
    safe_json_parse,
    to_json_string,
)

# XML结果添加策略的类型别名
XmlAddingStrategy = Literal["user_message", "assistant_message", "inline_edit"]


@dataclass
class ToolExecutionContext:
    """工具执行上下文，包含调用详情、结果和显示信息。"""

    tool_call: Dict[str, Any]
    tool_index: int
    result: Optional[ToolResult] = None
    function_name: Optional[str] = None
    xml_tag_name: Optional[str] = None
    error: Optional[Exception] = None
    assistant_message_id: Optional[str] = None
    parsing_details: Optional[Dict[str, Any]] = None


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

    async def _yield_message(
        self, message_obj: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """辅助方法：以适当的格式生成消息。

        确保内容和元数据为JSON字符串，以保证客户端兼容性。
        """
        if message_obj:
            return format_for_yield(message_obj)
        return None

    def _serialize_model_response(self, model_response) -> Dict[str, Any]:
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
            logger.warning(
                f"序列化ModelResponse失败: {str(e)}, 回退到字符串表示"
            )
            # 最终回退：转换为字符串
            return {"raw_response": str(model_response), "serialization_error": str(e)}

    async def _add_message_with_agent_info(
        self,
        thread_id: str,
        type: str,
        content: Union[Dict[str, Any], List[Any], str],
        is_llm_message: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
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
        prompt_messages: List[Dict[str, Any]],
        llm_model: str,
        config: ProcessorConfig = ProcessorConfig(),
        can_auto_continue: bool = False,
        auto_continue_count: int = 0,
        continuous_state: Optional[Dict[str, Any]] = None,
        estimated_total_tokens: Optional[int] = None,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """处理流式LLM响应，处理工具调用和执行。

        参数:
            llm_response: LLM的流式响应
            thread_id: 对话线程的ID
            prompt_messages: 发送给LLM的消息列表（提示）
            llm_model: 使用的LLM模型名称
            config: 解析和执行的配置
            can_auto_continue: 是否启用自动继续
            auto_continue_count: 自动继续的循环次数
            continuous_state: 对话的先前状态

        生成:
            完整的消息对象，匹配数据库模式，除了内容块。
        """
        logger.info(f"开始为线程 {thread_id} 处理流式响应")

        # 如果未提供，初始化取消事件
        if cancellation_event is None:
            cancellation_event = asyncio.Event()

        # 如果提供了continuous_state则从中初始化（用于自动继续）
        continuous_state = continuous_state or {}
        accumulated_content = continuous_state.get("accumulated_content", "")
        current_xml_content = accumulated_content  # 如果自动继续则等于accumulated_content，否则为空
        finish_reason = None
        should_auto_continue = False
        last_assistant_message_object = (
            None  # 存储最终保存的助手消息对象
        )
        has_printed_thinking_prefix = (
            False  # 仅打印一次思考前缀的标志
        )
        agent_should_terminate = (
            False  # 跟踪是否已执行终止工具的标志
        )
        complete_native_tool_calls = []  # 提前初始化，供assistant_response_end使用

        # 存储接收到的完整LiteLLM响应对象
        final_llm_response = None
        first_chunk_time = None
        last_chunk_time = None
        llm_response_end_saved = False

        # 重用thread_run_id用于自动继续或创建新的
        thread_run_id = continuous_state.get("thread_run_id") or str(uuid.uuid4())
        continuous_state["thread_run_id"] = thread_run_id

        # 关键：为本次特定的LLM调用生成唯一ID（不是每个线程运行）
        llm_response_id = str(uuid.uuid4())
        logger.info(
            f"🔵 LLM 调用 #{auto_continue_count + 1} 开始 - llm_response_id: {llm_response_id}"
        )

        try:
            # --- 保存并生成开始事件 ---
            if auto_continue_count == 0:
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
                "auto_continue_count": auto_continue_count,
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
                logger.info(
                    f"✅ 已保存第 #{auto_continue_count + 1} 次调用的 llm_response_start"
                )
            # --- 结束开始事件 ---

            __sequence = continuous_state.get(
                "sequence", 0
            )  # 从上一个自动继续循环中获取序列

            chunk_count = 0
            async for chunk in llm_response:
                # 处理每个块之前检查取消
                if cancellation_event.is_set():
                    logger.info(f"线程 {thread_id} 收到取消信号 - 停止 LLM 流处理")
                    finish_reason = "cancelled"
                    break

                chunk_count += 1

                # 跟踪时间
                current_time = datetime.now().timestamp()
                if first_chunk_time is None:
                    first_chunk_time = current_time
                last_chunk_time = current_time

                # 定期记录块信息以用于调试
                if (
                    chunk_count == 1
                    or (chunk_count % 1000 == 0)
                    or hasattr(chunk, "usage")
                ):
                    logger.debug(f"处理块 #{chunk_count}, 类型={type(chunk).__name__}")

                ## 当我们获得使用数据时，存储完整的LiteLLM响应块
                if (
                    hasattr(chunk, "usage")
                    and chunk.usage
                    and final_llm_response is None
                ):
                    logger.info(
                        "🔍 存储接收到的完整 LiteLLM 响应块"
                    )
                    final_llm_response = chunk  # 按原样存储整个块对象
                    logger.info(
                        f"🔍 存储的模型: {getattr(chunk, 'model', 'NO_MODEL')}"
                    )
                    logger.info(f"🔍 存储的使用量: {chunk.usage}")
                    logger.info(f"🔍 存储的响应类型: {type(chunk)}")

                if (
                    hasattr(chunk, "choices")
                    and chunk.choices
                    and hasattr(chunk.choices[0], "finish_reason")
                    and chunk.choices[0].finish_reason
                ):
                    finish_reason = chunk.choices[0].finish_reason
                    logger.debug(f"检测到 finish_reason：{finish_reason}")

                if hasattr(chunk, "choices") and chunk.choices:
                    delta = (
                        chunk.choices[0].delta
                        if hasattr(chunk.choices[0], "delta")
                        else None
                    )

                    # 检查并记录Anthropic的思考内容
                    if (
                        delta
                        and hasattr(delta, "reasoning_content")
                        and delta.reasoning_content
                    ):
                        if not has_printed_thinking_prefix:
                            # print("[THINKING]: ", end='', flush=True)
                            has_printed_thinking_prefix = True
                        # print(delta.reasoning_content, end='', flush=True)
                        # 将推理内容追加到主内容以保存在最终消息中
                        reasoning_content = delta.reasoning_content
                        # logger.debug(f"处理 reasoning_content: 类型={type(reasoning_content)}, 值={reasoning_content}")
                        if isinstance(reasoning_content, list):
                            reasoning_content = "".join(
                                str(item) for item in reasoning_content
                            )
                        # logger.debug(f"即将连接 reasoning_content (类型={type(reasoning_content)}) 到 accumulated_content (类型={type(accumulated_content)})")
                        accumulated_content += reasoning_content

                    # 处理内容块
                    if delta and hasattr(delta, "content") and delta.content:
                        chunk_content = delta.content
                        # logger.debug(f"处理 chunk_content: 类型={type(chunk_content)}, 值={chunk_content}")
                        if isinstance(chunk_content, list):
                            chunk_content = "".join(str(item) for item in chunk_content)
                        # print(chunk_content, end='', flush=True)
                        # logger.debug(f"即将连接 chunk_content (类型={type(chunk_content)}) 到 accumulated_content (类型={type(accumulated_content)})")
                        accumulated_content += chunk_content
                        # logger.debug(f"即将连接 chunk_content (类型={type(chunk_content)}) 到 current_xml_content (类型={type(current_xml_content)})")
                        current_xml_content += chunk_content

                        # 仅生成内容块（不保存）
                        now_chunk = datetime.now()
                        yield {
                            "sequence": __sequence,
                            "message_id": None,
                            "thread_id": thread_id,
                            "type": "assistant",
                            "is_llm_message": True,
                            "content": to_json_string(
                                {"role": "assistant", "content": chunk_content}
                            ),
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

            logger.info(f"流处理完成。总块数：{chunk_count}")

            # 如果有时间数据，计算响应时间
            response_ms = None
            if first_chunk_time and last_chunk_time:
                response_ms = (last_chunk_time - first_chunk_time) * 1000

            # 记录我们捕获的内容
            if final_llm_response:
                logger.info("✅ 已捕获完整的 LiteLLM 响应对象")
                logger.info(
                    f"🔍 响应模型: {getattr(final_llm_response, 'model', 'NO_MODEL')}"
                )
                logger.info(
                    f"🔍 响应使用量: {getattr(final_llm_response, 'usage', 'NO_USAGE')}"
                )
            else:
                logger.warning(
                    "⚠️ 未从流式块中捕获完整的 LiteLLM 响应"
                )

            should_auto_continue = can_auto_continue and finish_reason == "length"

            # 如果用户停止（取消），则不保存部分响应
            # 但对于其他提前停止（如达到XML限制）要保存
            if (
                accumulated_content
                and not should_auto_continue
                and finish_reason != "cancelled"
            ):
                message_data = {  # 要保存在'content'中的字典
                    "role": "assistant",
                    "content": accumulated_content,
                    "tool_calls": complete_native_tool_calls or None,
                }

                last_assistant_message_object = await self._add_message_with_agent_info(
                    thread_id=thread_id,
                    type="assistant",
                    content=message_data,
                    is_llm_message=True,
                    metadata={"thread_run_id": thread_run_id},
                )

                if last_assistant_message_object:
                    # 生成完整保存的对象，仅为生成添加stream_status元数据
                    yield_metadata = ensure_dict(
                        last_assistant_message_object.get("metadata"), {}
                    )
                    yield_metadata["stream_status"] = "complete"
                    # 格式化消息以供生成
                    yield_message = last_assistant_message_object.copy()
                    yield_message["metadata"] = yield_metadata
                    yield format_for_yield(yield_message)
                else:
                    logger.error(
                        f"为线程 {thread_id} 保存最终助手消息失败"
                    )
                    # 保存并生成错误状态
                    err_content = {
                        "role": "system",
                        "status_type": "error",
                        "message": "保存最终助手消息失败",
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

            # --- 最终完成状态 ---
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

            # 检查在处理待处理工具后代理是否应该终止
            if agent_should_terminate:
                logger.debug(
                    "执行ask/complete工具后请求代理终止。停止进一步处理。"
                )

                # 设置finish_reason以指示终止
                finish_reason = "agent_terminated"

                # 保存并生成终止状态
                finish_content = {
                    "status_type": "finish",
                    "finish_reason": "agent_terminated",
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

                # 在终止前保存llm_response_end
                if last_assistant_message_object:
                    try:
                        # 使用接收到的完整LiteLLM响应对象
                        if final_llm_response:
                            logger.info(
                                "✅ 在终止前使用完整的LiteLLM响应进行llm_response_end"
                            )
                            # 按原样序列化完整的响应对象
                            llm_end_content = self._serialize_model_response(
                                final_llm_response
                            )

                            # 添加流式标志和响应时间（如果可用）
                            llm_end_content["streaming"] = True
                            if response_ms:
                                llm_end_content["response_ms"] = response_ms

                            # 对于流式响应，我们需要手动构建choices
                            # 因为流式块没有完整的消息结构
                            llm_end_content["choices"] = [
                                {
                                    "finish_reason": finish_reason or "stop",
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": accumulated_content,
                                        "tool_calls": complete_native_tool_calls
                                        or None,
                                    },
                                }
                            ]
                            llm_end_content["llm_response_id"] = llm_response_id
                        else:
                            logger.warning(
                                "⚠️ 没有可用的完整LiteLLM响应，跳过llm_response_end"
                            )
                            llm_end_content = None

                        # 只有在我们有内容时才保存
                        if llm_end_content:
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
                            # 生成到流中以实时更新上下文使用量
                            if llm_end_msg_obj:
                                yield format_for_yield(llm_end_msg_obj)
                        logger.info(
                            f"✅ 在第 #{auto_continue_count + 1} 次调用前终止时已保存llm_response_end"
                        )
                    except Exception as e:
                        logger.error(
                            f"在终止前保存llm_response_end时出错: {str(e)}"
                        )

                # 跳过所有剩余处理并转到finally块
                return

            # --- 保存并生成llm_response_end ---
            # 仅在非自动继续时保存llm_response_end（响应实际完成）
            if not should_auto_continue:
                if last_assistant_message_object:
                    try:
                        # 使用接收到的完整LiteLLM响应对象
                        if final_llm_response:
                            logger.info(
                                "✅ 在正常完成时使用完整的LiteLLM响应进行llm_response_end"
                            )

                            # 记录完整的响应对象以用于调试
                            logger.info(
                                f"🔍 完整响应对象: {final_llm_response}"
                            )
                            logger.info(
                                f"🔍 响应对象类型: {type(final_llm_response)}"
                            )
                            logger.info(
                                f"🔍 响应对象字典: {final_llm_response.__dict__ if hasattr(final_llm_response, '__dict__') else 'NO_DICT'}"
                            )

                            # 按原样序列化完整的响应对象
                            llm_end_content = self._serialize_model_response(
                                final_llm_response
                            )
                            logger.info(f"🔍 序列化内容: {llm_end_content}")

                            # 添加流式标志和响应时间（如果可用）
                            llm_end_content["streaming"] = True
                            if response_ms:
                                llm_end_content["response_ms"] = response_ms

                            # 对于流式响应，我们需要手动构建choices
                            # 因为流式块没有完整的消息结构
                            llm_end_content["choices"] = [
                                {
                                    "finish_reason": finish_reason or "stop",
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": accumulated_content,
                                        "tool_calls": complete_native_tool_calls
                                        or None,
                                    },
                                }
                            ]
                            llm_end_content["llm_response_id"] = llm_response_id

                            # 调试：记录实际响应使用量
                            logger.info(
                                f"🔍 响应处理器完成使用量 (正常): {llm_end_content.get('usage', 'NO_USAGE')}"
                            )
                            logger.info(f"🔍 最终LLM结束内容: {llm_end_content}")

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
                            # 生成到流中以实时更新上下文使用量
                            if llm_end_msg_obj:
                                yield format_for_yield(llm_end_msg_obj)
                        else:
                            logger.warning(
                                "⚠️ 没有可用的完整LiteLLM响应，跳过llm_response_end"
                            )
                        logger.info(
                            f"✅ 在第 #{auto_continue_count + 1} 次调用正常完成时已保存llm_response_end"
                        )
                    except Exception as e:
                        logger.error(f"保存llm_response_end时出错: {str(e)}")

        except Exception as e:
            # 使用ErrorProcessor进行一致的错误处理
            processed_error = ErrorProcessor.process_system_error(
                e, context={"thread_id": thread_id}
            )
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
                metadata={
                    "thread_run_id": thread_run_id
                    if "thread_run_id" in locals()
                    else None
                },
            )
            if err_msg_obj:
                yield format_for_yield(err_msg_obj)
            raise

        finally:
            # 重要：finally块即使在流停止时也会运行（GeneratorExit）
            # 我们绝不能在这里生成 - 只需静默保存到数据库以用于计费/使用跟踪

            # 阶段3：资源清理 - 取消待处理任务并关闭生成器
            try:
                # 如果LLM响应生成器支持aclose()，则尝试关闭它
                # 这有助于阻止底层HTTP连接继续
                if hasattr(llm_response, "aclose"):
                    try:
                        await llm_response.aclose()
                        logger.debug(
                            f"已关闭线程 {thread_id} 的LLM响应生成器"
                        )
                    except Exception as close_err:
                        logger.debug(
                            f"关闭LLM响应生成器时出错（可能不支持aclose）: {close_err}"
                        )
                elif hasattr(llm_response, "close") and callable(getattr(llm_response, "close")):
                    try:
                        llm_response.close()  # type: ignore
                        logger.debug(
                            f"已关闭线程 {thread_id} 的LLM响应生成器（同步关闭）"
                        )
                    except Exception as close_err:
                        logger.debug(
                            f"关闭LLM响应生成器时出错（同步）: {close_err}"
                        )
            except Exception as cleanup_err:
                logger.warning(f"资源清理期间出错: {cleanup_err}")

            if not llm_response_end_saved and last_assistant_message_object:
                try:
                    logger.info(
                        f"💰 防弹计费：在finally块中为第 #{auto_continue_count + 1} 次调用保存llm_response_end"
                    )
                    if final_llm_response:
                        logger.info("💰 使用LLM响应的精确使用量")
                        llm_end_content = self._serialize_model_response(
                            final_llm_response
                        )
                    else:
                        logger.warning(
                            "💰 没有LLM响应使用量 - 为计费估算token使用量"
                        )
                        llm_end_content = {"model": llm_model, "usage": {}}

                    llm_end_content["streaming"] = True
                    llm_end_content["llm_response_id"] = llm_response_id

                    response_ms = None
                    if first_chunk_time and last_chunk_time:
                        response_ms = int((last_chunk_time - first_chunk_time) * 1000)
                        llm_end_content["response_ms"] = response_ms

                    llm_end_content["choices"] = [
                        {
                            "finish_reason": finish_reason or "interrupted",
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": accumulated_content,
                                "tool_calls": complete_native_tool_calls or None,
                            },
                        }
                    ]

                    usage_info = llm_end_content.get("usage", {})
                    is_estimated = usage_info.get("estimated", False)
                    logger.info(
                        f"💰 计费恢复 - 使用量 ({'估算' if is_estimated else '精确'}): {usage_info}"
                    )

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
                    # 不要在finally块中生成 - 流可能已关闭（GeneratorExit）
                    # 前端已停止消费，生成没有意义
                    logger.info(
                        f"✅ 计费成功：在finally中为第 #{auto_continue_count + 1} 次调用保存llm_response_end（{'估算' if is_estimated else '精确'}使用量）"
                    )

                except Exception as billing_e:
                    logger.error(
                        f"❌ 关键计费失败：无法保存llm_response_end: {str(billing_e)}",
                        exc_info=True,
                    )

            elif llm_response_end_saved:
                logger.debug(
                    f"✅ 第 #{auto_continue_count + 1} 次调用的计费已处理（llm_response_end已提前保存）"
                )

            if should_auto_continue:
                continuous_state["accumulated_content"] = accumulated_content
                continuous_state["sequence"] = __sequence

                logger.debug(
                    f"使用 {len(accumulated_content)} 个字符更新自动继续的持续状态"
                )
            else:
                # 保存并生成最终的thread_run_end状态（仅在非自动继续且finish_reason不是'length'时）
                try:
                    # 在元数据中存储last_usage以用于快速路径优化
                    usage = (
                        final_llm_response.usage
                        if "final_llm_response" in locals()
                        and final_llm_response is not None
                        and hasattr(final_llm_response, "usage")
                        else None
                    )

                    # 如果没有精确使用量（流提前停止），使用在thread_manager中预先计算的estimated_total
                    if not usage and estimated_total_tokens:
                        # 重用我们已在thread_manager中计算的estimated_total（无需数据库调用！）
                        class EstimatedUsage:
                            def __init__(self, total):
                                self.total_tokens = total

                        usage = EstimatedUsage(estimated_total_tokens)
                        logger.info(
                            f"⚡ 使用快速检查估算: {estimated_total_tokens} 个token（流已停止，无需重新计算）"
                        )

                    end_content = {"status_type": "thread_run_end"}

                    end_msg_obj = await self.add_message(
                        thread_id=thread_id,
                        type="status",
                        content=end_content,
                        is_llm_message=False,
                        metadata={
                            "thread_run_id": thread_run_id
                            if "thread_run_id" in locals()
                            else None
                        },
                    )
                    # 不要在finally块中生成 - 流可能已关闭（GeneratorExit）
                    logger.debug(
                        "在finally中保存thread_run_end（不生成以避免GeneratorExit）"
                    )
                except Exception as final_e:
                    logger.error(
                        f"finally块中出错: {str(final_e)}", exc_info=True
                    )

    async def process_non_streaming_response(
        self,
        llm_response: Any,
        thread_id: str,
        prompt_messages: List[Dict[str, Any]],
        llm_model: str,
        config: ProcessorConfig = ProcessorConfig(),
        estimated_total_tokens: Optional[int] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
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
                    llm_response.choices[0].message
                    if hasattr(llm_response.choices[0], "message")
                    else None
                )
                if response_message:
                    if (
                        hasattr(response_message, "content")
                        and response_message.content
                    ):
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
                logger.error(
                    f"为线程 {thread_id} 保存非流式助手消息失败"
                )
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
                    logger.error(
                        f"为非流式保存助手响应结束时出错: {str(e)}"
                    )

        except Exception as e:
            # 使用ErrorProcessor进行一致的错误处理
            processed_error = ErrorProcessor.process_system_error(
                e, context={"thread_id": thread_id}
            )
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
                metadata={
                    "thread_run_id": thread_run_id
                    if "thread_run_id" in locals()
                    else None
                },
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
                metadata={
                    "thread_run_id": thread_run_id
                    if "thread_run_id" in locals()
                    else None
                },
            )
            if end_msg_obj:
                yield format_for_yield(end_msg_obj)

    async def _add_tool_result(
        self,
        thread_id: str,
        tool_call: Dict[str, Any],
        result: ToolResult,
        strategy: Union[XmlAddingStrategy, str] = "assistant_message",
        assistant_message_id: Optional[str] = None,
        parsing_details: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:  # 返回完整的消息对象
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
            parsing_details: XML调用的可选解析详情（属性、元素等）
        """
        try:
            message_obj = None  # 初始化message_obj

            # 如果提供了assistant_message_id，则创建包含它的元数据
            metadata = {}
            if assistant_message_id:
                metadata["assistant_message_id"] = assistant_message_id
                logger.debug(
                    f"将工具结果链接到助手消息: {assistant_message_id}"
                )

            # --- 如果可用，将解析详情添加到元数据 ---
            if parsing_details:
                metadata["parsing_details"] = parsing_details
                logger.debug("将parsing_details添加到工具结果元数据")
            # ---

            # 检查这是否是原生函数调用（具有id字段）
            if "id" in tool_call:
                # 根据OpenAI规范格式化为适当的工具消息
                function_name = tool_call.get("function_name", "")

                # 格式化工具结果内容 - 工具角色需要字符串内容
                if isinstance(result, str):
                    content = result
                elif hasattr(result, "output"):
                    # 如果是ToolResult对象
                    if isinstance(result.output, dict) or isinstance(
                        result.output, list
                    ):
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

                logger.debug(
                    f"为tool_call_id={tool_call['id']}添加原生工具结果，角色为tool"
                )

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
                tool_call, result, parsing_details, for_llm=False
            )
            # 2. 用于LLM的简洁版本
            structured_result_for_llm = self._create_structured_tool_result(
                tool_call, result, parsing_details, for_llm=True
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
                    metadata={"assistant_message_id": assistant_message_id}
                    if assistant_message_id
                    else {},
                )
                return message_obj  # 返回完整的消息对象
            except Exception as e2:
                logger.error(
                    f"即使使用回退消息也失败: {str(e2)}", exc_info=True
                )
                return None  # 出错时返回None

    def _create_structured_tool_result(
        self,
        tool_call: Dict[str, Any],
        result: ToolResult,
        parsing_details: Optional[Dict[str, Any]] = None,
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
        xml_tag_name = tool_call.get("xml_tag_name")
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
                "xml_tag_name": xml_tag_name,
                "tool_call_id": tool_call_id,
                "arguments": arguments,
                "result": {
                    "success": result.success if hasattr(result, "success") else True,
                    "output": output,
                    "error": getattr(result, "error", None)
                    if hasattr(result, "error")
                    else None,
                },
            }
        }

        return structured_result_v1

    def _create_tool_context(
        self,
        tool_call: Dict[str, Any],
        tool_index: int,
        assistant_message_id: Optional[str] = None,
        parsing_details: Optional[Dict[str, Any]] = None,
    ) -> ToolExecutionContext:
        """创建包含显示名称和解析详情的工具执行上下文。"""
        context = ToolExecutionContext(
            tool_call=tool_call,
            tool_index=tool_index,
            assistant_message_id=assistant_message_id,
            parsing_details=parsing_details,
        )

        # 设置function_name和xml_tag_name字段
        if "xml_tag_name" in tool_call:
            context.xml_tag_name = tool_call["xml_tag_name"]
            context.function_name = tool_call.get(
                "function_name", tool_call["xml_tag_name"]
            )
        else:
            # 对于非XML工具，直接使用函数名
            context.function_name = tool_call.get("function_name", "unknown")
            context.xml_tag_name = None

        return context

    async def _yield_and_save_tool_started(
        self, context: ToolExecutionContext, thread_id: str, thread_run_id: str
    ) -> Optional[Dict[str, Any]]:
        """格式化、保存并返回工具开始状态消息。"""
        tool_name = context.xml_tag_name or context.function_name
        content = {
            "role": "assistant",
            "status_type": "tool_started",
            "function_name": context.function_name,
            "xml_tag_name": context.xml_tag_name,
            "message": f"开始执行 {tool_name}",
            "tool_index": context.tool_index,
            "tool_call_id": context.tool_call.get(
                "id"
            ),  # 如果是原生的，包含tool_call ID
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
    ) -> Optional[Dict[str, Any]]:
        """格式化、保存并返回工具完成/失败状态消息。"""
        if not context.result:
            # 如果结果缺失（例如执行失败），委托给错误保存
            return await self._yield_and_save_tool_error(
                context, thread_id, thread_run_id
            )

        tool_name = context.xml_tag_name or context.function_name
        status_type = "tool_completed" if context.result.success else "tool_failed"
        message_text = f"工具 {tool_name} {'成功完成' if context.result.success else '失败'}"

        content = {
            "role": "assistant",
            "status_type": status_type,
            "function_name": context.function_name,
            "xml_tag_name": context.xml_tag_name,
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
            logger.debug(
                f"使用终止信号标记工具状态 '{context.function_name}'。"
            )
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
    ) -> Optional[Dict[str, Any]]:
        """格式化、保存并返回工具错误状态消息。"""
        error_msg = (
            str(context.error)
            if context.error
            else "工具执行期间未知错误"
        )
        tool_name = context.xml_tag_name or context.function_name
        content = {
            "role": "assistant",
            "status_type": "tool_error",
            "function_name": context.function_name,
            "xml_tag_name": context.xml_tag_name,
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
