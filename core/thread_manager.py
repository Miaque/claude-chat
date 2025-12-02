import asyncio
import json
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any, Literal, Optional, Union, cast

from claude_agent_sdk.types import PermissionMode
from loguru import logger
from sqlalchemy import select

from core.error_processor import ErrorProcessor
from core.response_processor import ProcessorConfig, ResponseProcessor
from core.services.db import get_db
from core.services.llm import LLMError, make_llm_api_call
from core.tool import Tool
from models.message import Message, Messages
from models.thread import Thread, Threads

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
        tool_class: type[Tool],
        function_names: Optional[list[str]] = None,
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
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """在数据库中创建新线程。"""
        # logger.debug(f"创建新线程 (account_id: {account_id}, project_id: {project_id})")

        thread_data = {"is_public": is_public, "metadata": metadata or {}}
        if account_id:
            thread_data["account_id"] = account_id
        if project_id:
            thread_data["project_id"] = project_id

        thread = Threads.insert(Thread(**thread_data))
        thread_id = thread.thread_id
        logger.info("成功创建线程: {}", thread_id)
        return thread_id

    async def add_message(
        self,
        thread_id: str,
        type: str,
        content: Union[dict[str, Any], list[Any], str],
        is_llm_message: bool = False,
        metadata: Optional[dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        agent_version_id: Optional[str] = None,
    ):
        """向线程中添加消息到数据库。"""
        # logger.debug(f"向线程 {thread_id} 添加类型为 '{type}' 的消息")

        current_time = datetime.now()
        data_to_insert = {
            "thread_id": thread_id,
            "type": type,
            "content": content,
            "is_llm_message": is_llm_message,
            "meta": metadata or {},
            "created_at": current_time,
            "updated_at": current_time,
        }

        if agent_id:
            data_to_insert["agent_id"] = agent_id
        if agent_version_id:
            data_to_insert["agent_version_id"] = agent_version_id

        message = Message(**data_to_insert)
        saved_message = Messages.insert(message)
        return saved_message.model_dump(mode="json")

    async def get_llm_messages(self, thread_id: str) -> list[dict[str, Any]]:
        """获取线程的所有消息。"""
        logger.debug(f"获取线程 {thread_id} 的消息")

        try:
            all_messages = []
            batch_size = 1000
            offset = 0

            while True:
                with get_db() as db:
                    result = (
                        db.execute(
                            select(
                                Message.message_id,
                                Message.type,
                                Message.content,
                                Message.meta,
                            )
                            .filter(Message.thread_id == thread_id)
                            .filter(Message.is_llm_message)
                            .order_by(Message.created_at)
                            .offset(offset)
                            .limit(batch_size)
                        )
                        .mappings()
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
                metadata = item.get("meta", {})

                # 解析内容并添加message_id
                if isinstance(content, str):
                    try:
                        parsed_item = json.loads(content)
                        parsed_item["message_id"] = item["message_id"]
                        messages.append(parsed_item)
                    except json.JSONDecodeError:
                        logger.error(f"解析消息失败: {content[:100]}")
                else:
                    content["message_id"] = str(item["message_id"])
                    messages.append(content)

            return messages

        except Exception as e:
            logger.exception(f"获取线程 {thread_id} 的消息失败")
            return []

    async def run_thread(
        self,
        thread_id: str,
        system_prompt: dict[str, Any],
        stream: bool = True,
        permission_mode: PermissionMode | None = None,
        temporary_message: Optional[dict[str, Any]] = None,
        llm_model: str = "glm-4.6",
        processor_config: Optional[ProcessorConfig] = None,
        tool_choice: ToolChoice = "auto",
        latest_user_message_content: Optional[str] = None,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> Union[dict[str, Any], AsyncGenerator]:
        """运行对话线程，集成LLM和工具执行。"""
        logger.debug(f"开始执行线程 {thread_id}，使用模型 {llm_model}")

        # 确保ProcessorConfig对象有效
        if processor_config is None:
            config = ProcessorConfig()
        elif isinstance(processor_config, ProcessorConfig):
            config = processor_config
        else:
            logger.error(f"无效的processor_config类型: {type(processor_config)}，创建默认值")
            config = ProcessorConfig()

        result = await self._execute_run(
            thread_id,
            system_prompt,
            llm_model,
            tool_choice,
            config,
            stream,
            permission_mode,
            temporary_message,
            latest_user_message_content,
            cancellation_event,
        )

        # 如果结果是错误字典，将其转换为生成器并产出错误
        if isinstance(result, dict) and result.get("status") == "error":
            return self._create_single_error_generator(result)

        return result

    async def _execute_run(
        self,
        thread_id: str,
        system_prompt: dict[str, Any],
        llm_model: str,
        tool_choice: ToolChoice,
        config: ProcessorConfig,
        stream: bool,
        permission_mode: PermissionMode | None = None,
        temporary_message: Optional[dict[str, Any]] = None,
        latest_user_message_content: Optional[str] = None,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> Union[dict[str, Any], AsyncGenerator]:
        """执行单次LLM运行。"""

        # 关键: 确保config始终是ProcessorConfig对象
        if not isinstance(config, ProcessorConfig):
            logger.error(f"错误: config是{type(config)}，期望ProcessorConfig。值: {config}")
            config = ProcessorConfig()  # 创建新实例作为后备

        try:
            messages = await self.get_llm_messages(thread_id)
            thread = Threads.get_by_id(thread_id, Thread.session_id)
            session_id = thread.session_id if thread else None

            # 获取LLM调用的工具模式(在压缩之后)
            openapi_tool_schemas = None

            prepared_messages = messages

            # 注意: 我们不在此处记录token计数，因为缓存块给出不准确的计数
            # LLM的usage.prompt_tokens(在调用后报告)是准确的真相来源
            logger.info(f"向LLM发送 {len(prepared_messages)} 条准备好的消息")

            # 调用LLM
            try:
                llm_response = await make_llm_api_call(
                    prepared_messages,
                    llm_model,
                    tools=openapi_tool_schemas,
                    stream=stream,
                    permission_mode=permission_mode,
                    system_prompt=system_prompt.get("content"),
                    session_id=session_id,
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
                    cancellation_event,
                )
            else:
                return self.response_processor.process_non_streaming_response(
                    llm_response,
                    thread_id,
                    prepared_messages,
                    llm_model,
                    config,
                )

        except Exception as e:
            processed_error = ErrorProcessor.process_system_error(e, context={"thread_id": thread_id})
            ErrorProcessor.log_error(processed_error)
            return processed_error.to_stream_dict()

    async def _create_single_error_generator(self, error_dict: dict[str, Any]):
        """创建产出单个错误消息的异步生成器。"""
        yield error_dict
