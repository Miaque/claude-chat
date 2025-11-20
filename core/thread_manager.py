import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Type, Union, cast

from loguru import logger

from core.error_processor import ErrorProcessor
from core.response_processor import ProcessorConfig, ResponseProcessor
from core.services.db import get_db
from core.services.llm import LLMError, make_llm_api_call
from core.tool import Tool
from models.message import Message, MessageModel
from models.thread import Thread

ToolChoice = Literal["auto", "required", "none"]


class ThreadManager:
    """管理对话线程，集成LLM模型和工具执行。"""

    def __init__(
        self,
        agent_config: Optional[dict] = None,
    ):
        # self.tool_registry = ToolRegistry()

        self.agent_config = agent_config
        self.response_processor = ResponseProcessor(
            add_message_callback=self.add_message,
            agent_config=self.agent_config,
        )

    def add_tool(
        self,
        tool_class: Type[Tool],
        function_names: Optional[List[str]] = None,
        **kwargs,
    ):
        """向ThreadManager添加工具。"""
        # self.tool_registry.register_tool(tool_class, function_names, **kwargs)
        pass

    async def create_thread(
        self,
        account_id: Optional[str] = None,
        project_id: Optional[str] = None,
        is_public: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """在数据库中创建新线程。"""
        # logger.debug(f"创建新线程 (account_id: {account_id}, project_id: {project_id})")

        thread_data = {"is_public": is_public, "metadata": metadata or {}}
        if account_id:
            thread_data["account_id"] = account_id
        if project_id:
            thread_data["project_id"] = project_id

        try:
            with get_db() as db:
                thread = Thread(**thread_data)
                db.add(thread)
                db.commit()
                db.refresh(thread)

            if thread:
                thread_id = thread.thread_id
                logger.info(f"成功创建线程: {thread_id}")
                return thread_id
            else:
                raise Exception("创建线程失败: 未返回thread_id")
        except Exception as e:
            logger.error(f"创建线程失败: {str(e)}", exc_info=True)
            raise Exception(f"线程创建失败: {str(e)}")

    async def add_message(
        self,
        thread_id: str,
        type: str,
        content: Union[Dict[str, Any], List[Any], str],
        is_llm_message: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        agent_version_id: Optional[str] = None,
    ):
        """向线程中添加消息到数据库。"""
        # logger.debug(f"向线程 {thread_id} 添加类型为 '{type}' 的消息")

        data_to_insert = {
            "thread_id": thread_id,
            "type": type,
            "content": content,
            "is_llm_message": is_llm_message,
            "metadata": metadata or {},
        }

        if agent_id:
            data_to_insert["agent_id"] = agent_id
        if agent_version_id:
            data_to_insert["agent_version_id"] = agent_version_id

        try:
            with get_db() as db:
                message = Message(**data_to_insert)
                db.add(message)
                db.commit()
                db.refresh(message)

            if message:
                saved_message = MessageModel.model_validate(message)

                return saved_message
            else:
                logger.error(f"线程 {thread_id} 的插入操作失败")
                return None
        except Exception as e:
            logger.error(f"向线程 {thread_id} 添加消息失败: {str(e)}", exc_info=True)
            raise

    async def get_llm_messages(self, thread_id: str) -> List[Dict[str, Any]]:
        """获取线程的所有消息。"""
        logger.debug(f"获取线程 {thread_id} 的消息")

        try:
            all_messages = []
            batch_size = 1000
            offset = 0

            while True:
                with get_db() as db:
                    result = (
                        db.query(
                            Message.message_id,
                            Message.type,
                            Message.content,
                            Message.metadata,
                        )
                        .filter(Message.thread_id == thread_id)
                        .filter(Message.is_llm_message == True)
                        .order_by(Message.created_at)
                        .offset(offset)
                        .limit(batch_size)
                        .all()
                    )

                if not result:
                    break

                all_messages.extend(result)
                if len(result) < batch_size:
                    break
                offset += batch_size

            if not all_messages:
                return []

            messages = []
            for item in all_messages:
                # 检查此消息在元数据中是否有压缩版本
                content = item["content"]
                metadata = item.get("metadata", {})
                is_compressed = False

                # 如果已压缩，对LLM使用compressed_content而不是完整内容
                if isinstance(metadata, dict) and metadata.get("compressed"):
                    compressed_content = metadata.get("compressed_content")
                    if compressed_content:
                        content = compressed_content
                        is_compressed = True
                        # logger.debug(f"对消息 {item['message_id']} 使用压缩内容")

                # 解析内容并添加message_id
                if isinstance(content, str):
                    try:
                        parsed_item = json.loads(content)
                        parsed_item["message_id"] = item["message_id"]
                        messages.append(parsed_item)
                    except json.JSONDecodeError:
                        # 如果已压缩，内容是纯字符串(不是JSON) - 这是预期的
                        if is_compressed:
                            messages.append(
                                {
                                    "role": "user",
                                    "content": content,
                                    "message_id": item["message_id"],
                                }
                            )
                        else:
                            logger.error(f"解析消息失败: {content[:100]}")
                else:
                    content["message_id"] = item["message_id"]
                    messages.append(content)

            return messages

        except Exception as e:
            logger.error(
                f"获取线程 {thread_id} 的消息失败: {str(e)}",
                exc_info=True,
            )
            return []

    async def run_thread(
        self,
        thread_id: str,
        system_prompt: Dict[str, Any],
        stream: bool = True,
        temporary_message: Optional[Dict[str, Any]] = None,
        llm_model: str = "gpt-5",
        llm_temperature: float = 0,
        llm_max_tokens: Optional[int] = None,
        processor_config: Optional[ProcessorConfig] = None,
        tool_choice: ToolChoice = "auto",
        native_max_auto_continues: int = 25,
        max_xml_tool_calls: int = 0,
        latest_user_message_content: Optional[str] = None,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> Union[Dict[str, Any], AsyncGenerator]:
        """运行对话线程，集成LLM和工具执行。"""
        logger.debug(f"🚀 开始执行线程 {thread_id}，使用模型 {llm_model}")

        # 确保我们有有效的ProcessorConfig对象
        if processor_config is None:
            config = ProcessorConfig()
        elif isinstance(processor_config, ProcessorConfig):
            config = processor_config
        else:
            logger.error(
                f"无效的processor_config类型: {type(processor_config)}，创建默认值"
            )
            config = ProcessorConfig()

        auto_continue_state = {
            "count": 0,
            "active": True,
            "continuous_state": {"accumulated_content": "", "thread_run_id": None},
        }

        # 如果禁用自动继续，则单次执行
        if native_max_auto_continues == 0:
            result = await self._execute_run(
                thread_id,
                system_prompt,
                llm_model,
                llm_temperature,
                llm_max_tokens,
                tool_choice,
                config,
                stream,
                auto_continue_state,
                temporary_message,
                latest_user_message_content,
                cancellation_event,
            )

            # 如果结果是错误字典，将其转换为生成器并产出错误
            if isinstance(result, dict) and result.get("status") == "error":
                return self._create_single_error_generator(result)

            return result

        # 自动继续执行
        return self._auto_continue_generator(
            thread_id,
            system_prompt,
            llm_model,
            llm_temperature,
            llm_max_tokens,
            tool_choice,
            config,
            stream,
            auto_continue_state,
            temporary_message,
            native_max_auto_continues,
            latest_user_message_content,
            cancellation_event,
        )

    async def _execute_run(
        self,
        thread_id: str,
        system_prompt: Dict[str, Any],
        llm_model: str,
        llm_temperature: float,
        llm_max_tokens: Optional[int],
        tool_choice: ToolChoice,
        config: ProcessorConfig,
        stream: bool,
        auto_continue_state: Dict[str, Any],
        temporary_message: Optional[Dict[str, Any]] = None,
        latest_user_message_content: Optional[str] = None,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> Union[Dict[str, Any], AsyncGenerator]:
        """执行单次LLM运行。"""

        # 关键: 确保config始终是ProcessorConfig对象
        if not isinstance(config, ProcessorConfig):
            logger.error(
                f"错误: config是{type(config)}，期望ProcessorConfig。值: {config}"
            )
            config = ProcessorConfig()  # 创建新实例作为后备

        try:
            estimated_total_tokens = None  # 将传递给响应处理器以避免重新计算

            # 关键: 首先检查这是否是自动继续迭代(在任何token计数之前)
            is_auto_continue = auto_continue_state.get("count", 0) > 0

            # 始终获取消息(需要用于LLM调用)
            # 快速路径只是跳过压缩，而不是获取！
            messages = await self.get_llm_messages(thread_id)

            # 处理自动继续上下文
            if auto_continue_state["count"] > 0 and auto_continue_state[
                "continuous_state"
            ].get("accumulated_content"):
                partial_content = auto_continue_state["continuous_state"][
                    "accumulated_content"
                ]
                messages.append({"role": "assistant", "content": partial_content})

            # 获取LLM调用的工具模式(在压缩之后)
            openapi_tool_schemas = None

            prepared_messages = messages

            # 注意: 我们不在此处记录token计数，因为缓存块给出不准确的计数
            # LLM的usage.prompt_tokens(在调用后报告)是准确的真相来源
            logger.info(f"📤 向LLM发送 {len(prepared_messages)} 条准备好的消息")

            # 调用LLM
            try:
                llm_response = await make_llm_api_call(
                    prepared_messages,
                    llm_model,
                    temperature=llm_temperature,
                    max_tokens=llm_max_tokens,
                    tools=openapi_tool_schemas,
                    stream=stream,
                )
            except LLMError as e:
                return {"type": "status", "status": "error", "message": str(e)}

            # 检查错误响应
            if isinstance(llm_response, dict) and llm_response.get("status") == "error":
                return llm_response

            if stream and hasattr(llm_response, "__aiter__"):
                return self.response_processor.process_streaming_response(
                    cast(AsyncGenerator, llm_response),
                    thread_id,
                    prepared_messages,
                    llm_model,
                    config,
                    True,
                    auto_continue_state["count"],
                    auto_continue_state["continuous_state"],
                    estimated_total_tokens,
                    cancellation_event,
                )
            else:
                return self.response_processor.process_non_streaming_response(
                    llm_response,
                    thread_id,
                    prepared_messages,
                    llm_model,
                    config,
                    estimated_total_tokens,
                )

        except Exception as e:
            processed_error = ErrorProcessor.process_system_error(
                e, context={"thread_id": thread_id}
            )
            ErrorProcessor.log_error(processed_error)
            return processed_error.to_stream_dict()

    async def _auto_continue_generator(
        self,
        thread_id: str,
        system_prompt: Dict[str, Any],
        llm_model: str,
        llm_temperature: float,
        llm_max_tokens: Optional[int],
        tool_choice: ToolChoice,
        config: ProcessorConfig,
        stream: bool,
        auto_continue_state: Dict[str, Any],
        temporary_message: Optional[Dict[str, Any]],
        native_max_auto_continues: int,
        latest_user_message_content: Optional[str] = None,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator:
        """处理自动继续逻辑的生成器。"""
        logger.debug(f"启动自动继续生成器，最大次数: {native_max_auto_continues}")
        # logger.debug(f"自动继续生成器中的Config类型: {type(config)}")

        # 确保config是有效的ProcessorConfig
        if not isinstance(config, ProcessorConfig):
            logger.error(f"自动继续中无效的config类型: {type(config)}，创建新的")
            config = ProcessorConfig()

        while (
            auto_continue_state["active"]
            and auto_continue_state["count"] < native_max_auto_continues
        ):
            auto_continue_state["active"] = False  # 重置本次迭代

            try:
                # 继续前检查取消信号
                if cancellation_event and cancellation_event.is_set():
                    logger.info(f"线程 {thread_id} 的自动继续生成器收到取消信号")
                    break

                response_gen = await self._execute_run(
                    thread_id,
                    system_prompt,
                    llm_model,
                    llm_temperature,
                    llm_max_tokens,
                    tool_choice,
                    config,
                    stream,
                    auto_continue_state,
                    temporary_message if auto_continue_state["count"] == 0 else None,
                    latest_user_message_content
                    if auto_continue_state["count"] == 0
                    else None,
                    cancellation_event,
                )

                # 处理错误响应
                if (
                    isinstance(response_gen, dict)
                    and response_gen.get("status") == "error"
                ):
                    yield response_gen
                    break

                # 处理流式响应
                if hasattr(response_gen, "__aiter__"):
                    async for chunk in cast(AsyncGenerator, response_gen):
                        # 检查取消信号
                        if cancellation_event and cancellation_event.is_set():
                            logger.info(
                                f"处理线程 {thread_id} 自动继续流时收到取消信号"
                            )
                            break

                        # 检查自动继续触发器
                        should_continue = self._check_auto_continue_trigger(
                            chunk, auto_continue_state, native_max_auto_continues
                        )

                        # 跳过触发自动继续的完成块(但不是工具执行，前端需要那些)
                        if should_continue:
                            if chunk.get("type") == "status":
                                try:
                                    content = json.loads(chunk.get("content", "{}"))
                                    # 仅跳过长度限制完成状态(前端需要工具执行完成)
                                    if content.get("finish_reason") == "length":
                                        continue
                                except (json.JSONDecodeError, TypeError):
                                    pass

                        yield chunk
                else:
                    yield response_gen

                if not auto_continue_state["active"]:
                    break

            except Exception as e:
                processed_error = ErrorProcessor.process_system_error(
                    e, context={"thread_id": thread_id}
                )
                ErrorProcessor.log_error(processed_error)
                yield processed_error.to_stream_dict()
                return

        # 处理达到最大迭代次数
        if (
            auto_continue_state["active"]
            and auto_continue_state["count"] >= native_max_auto_continues
        ):
            logger.warning(f"达到最大自动继续限制 ({native_max_auto_continues})")
            yield {
                "type": "content",
                "content": f"\n[Agent达到最大自动继续限制 {native_max_auto_continues}]",
            }

    def _check_auto_continue_trigger(
        self,
        chunk: Dict[str, Any],
        auto_continue_state: Dict[str, Any],
        native_max_auto_continues: int,
    ) -> bool:
        """检查响应块是否应该触发自动继续。"""
        if chunk.get("type") == "status":
            try:
                content = (
                    json.loads(chunk.get("content", "{}"))
                    if isinstance(chunk.get("content"), str)
                    else chunk.get("content", {})
                )
                finish_reason = content.get("finish_reason")
                tools_executed = content.get("tools_executed", False)

                # 为以下情况触发自动继续: 原生工具调用、长度限制或XML工具已执行
                if finish_reason == "tool_calls" or tools_executed:
                    if native_max_auto_continues > 0:
                        logger.debug(
                            f"因工具执行自动继续 ({auto_continue_state['count'] + 1}/{native_max_auto_continues})"
                        )
                        auto_continue_state["active"] = True
                        auto_continue_state["count"] += 1
                        return True
                elif finish_reason == "length":
                    logger.debug(
                        f"因长度限制自动继续 ({auto_continue_state['count'] + 1}/{native_max_auto_continues})"
                    )
                    auto_continue_state["active"] = True
                    auto_continue_state["count"] += 1
                    return True
                elif finish_reason == "xml_tool_limit_reached":
                    logger.debug("因XML工具限制停止自动继续")
                    auto_continue_state["active"] = False
            except (json.JSONDecodeError, TypeError):
                pass

        return False

    async def _create_single_error_generator(self, error_dict: Dict[str, Any]):
        """创建产出单个错误消息的异步生成器。"""
        yield error_dict
