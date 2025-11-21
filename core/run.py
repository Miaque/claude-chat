import asyncio
import datetime
import json
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, Optional

from loguru import logger

from core.error_processor import ErrorProcessor
from core.prompts.prompt import get_system_prompt
from core.response_processor import ProcessorConfig
from core.services.db import get_db
from core.thread_manager import ThreadManager
from models.message import Message
from models.project import Project, ProjectModel
from models.thread import Thread


@dataclass
class AgentConfig:
    thread_id: str
    project_id: str
    native_max_auto_continues: int = 25
    max_iterations: int = 100
    model_name: str = "glm-4.6"
    agent_config: Optional[dict] = None


class PromptManager:
    @staticmethod
    async def build_system_prompt(
        model_name: str,
        agent_config: Optional[dict],
        thread_id: str,
        tool_registry=None,
        user_id: Optional[str] = None,
    ) -> dict:
        default_system_content = get_system_prompt()

        # 从代理的正常系统提示或默认提示开始
        if agent_config and agent_config.get("system_prompt"):
            system_content = agent_config["system_prompt"].strip()
        else:
            system_content = default_system_content

        now = datetime.datetime.now(datetime.timezone.utc)
        datetime_info = f"\n\n=== 当前日期/时间信息 ===\n"
        datetime_info += f"今天的日期: {now.strftime('%A, %B %d, %Y')}\n"
        datetime_info += f"当前年份: {now.strftime('%Y')}\n"
        datetime_info += f"当前月份: {now.strftime('%B')}\n"
        datetime_info += f"当前日期: {now.strftime('%A')}\n"
        datetime_info += (
            "将此信息用于任何时间敏感的任务、研究，或需要当前日期/时间上下文时。\n"
        )

        system_content += datetime_info

        # 如果提供了user_id，添加用户地区上下文
        if user_id:
            try:
                from core.utils.user_locale import (
                    get_locale_context_prompt,
                    get_user_locale,
                )

                locale = await get_user_locale(user_id)
                locale_prompt = get_locale_context_prompt(locale)
                system_content += f"\n\n{locale_prompt}\n"
                logger.debug(
                    f"为用户 {user_id} 添加了地区上下文 ({locale}) 到系统提示中"
                )
            except Exception as e:
                logger.warning(f"向系统提示添加地区上下文失败: {e}")

        system_message = {"role": "system", "content": system_content}
        return system_message


class AgentRunner:
    def __init__(self, config: AgentConfig):
        self.config = config

    async def setup(self):
        self.thread_manager = ThreadManager(agent_config=self.config.agent_config)

        with get_db() as db:
            response = (
                db.query(Thread.account_id)
                .filter(Thread.thread_id == self.config.thread_id)
                .first()
            )

        if not response:
            raise ValueError(f"未找到线程 {self.config.thread_id}")

        self.account_id = response.account_id

        if not self.account_id:
            raise ValueError(f"线程 {self.config.thread_id} 没有关联的账户")

        with get_db() as db:
            project = (
                db.query(Project)
                .filter(Project.project_id == self.config.project_id)
                .first()
            )

        if not project:
            raise ValueError(f"未找到项目 {self.config.project_id}")

        project_data = ProjectModel.model_validate(project)
        sandbox_info = project_data.sandbox
        if not sandbox_info.get("id"):
            logger.debug(
                f"未找到项目 {self.config.project_id} 的sandbox；将在需要时延迟创建"
            )

    async def run(
        self, cancellation_event: Optional[asyncio.Event] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        await self.setup()

        system_message = await PromptManager.build_system_prompt(
            self.config.model_name,
            self.config.agent_config,
            self.config.thread_id,
            # tool_registry=self.thread_manager.tool_registry,
            user_id=self.account_id,
        )
        logger.info(
            f"📝 系统消息构建完成: {len(str(system_message.get('content', '')))} 字符"
        )
        logger.debug(f"收到 model_name: {self.config.model_name}")
        iteration_count = 0
        continue_execution = True

        with get_db() as db:
            latest_user_message = (
                db.query(Message)
                .filter(Message.thread_id == self.config.thread_id)
                .filter(Message.type == "user")
                .order_by(Message.created_at.desc())
                .first()
            )

        latest_user_message_content = None
        if latest_user_message:
            data = latest_user_message.content
            if isinstance(data, str):
                data = json.loads(data)
            # 提取内容用于快速路径优化
            latest_user_message_content = (
                data.get("content") if isinstance(data, dict) else str(data)
            )

        while continue_execution and iteration_count < self.config.max_iterations:
            iteration_count += 1

            with get_db() as db:
                latest_message = (
                    db.query(Message)
                    .filter(Message.thread_id == self.config.thread_id)
                    .filter(Message.type.in_(["assistant", "tool", "user"]))
                    .order_by(Message.created_at.desc())
                    .first()
                )

            if latest_message:
                message_type = latest_message.type
                if message_type == "assistant":
                    continue_execution = False
                    break

            temporary_message = None
            # 默认不设置max_tokens - 让LiteLLM和提供商处理自己的默认值
            max_tokens = None
            logger.debug(f"max_tokens: {max_tokens} (使用提供商默认值)")
            try:
                logger.debug(f"开始为 {self.config.thread_id} 执行线程")
                response = await self.thread_manager.run_thread(
                    thread_id=self.config.thread_id,
                    system_prompt=system_message,
                    stream=True,
                    llm_model=self.config.model_name,
                    llm_temperature=0,
                    llm_max_tokens=max_tokens,
                    tool_choice="auto",
                    max_xml_tool_calls=1,
                    temporary_message=temporary_message,
                    latest_user_message_content=latest_user_message_content,
                    processor_config=ProcessorConfig(
                        execute_on_stream=True,
                    ),
                    native_max_auto_continues=self.config.native_max_auto_continues,
                    cancellation_event=cancellation_event,
                )

                last_tool_call = None
                agent_should_terminate = False
                error_detected = False

                try:
                    if hasattr(response, "__aiter__") and not isinstance(
                        response, dict
                    ):
                        async for chunk in response:
                            # 检查来自thread_manager的错误状态
                            if (
                                isinstance(chunk, dict)
                                and chunk.get("type") == "status"
                                and chunk.get("status") == "error"
                            ):
                                logger.error(
                                    f"线程执行出错: {chunk.get('message', '未知错误')}"
                                )
                                error_detected = True
                                yield chunk
                                continue

                            # 检查流中的错误状态（消息格式）
                            if (
                                isinstance(chunk, dict)
                                and chunk.get("type") == "status"
                            ):
                                try:
                                    content = chunk.get("content", {})
                                    if isinstance(content, str):
                                        content = json.loads(content)

                                    # 检查错误状态
                                    if content.get("status_type") == "error":
                                        error_detected = True
                                        yield chunk
                                        continue

                                    # 检查代理终止
                                    metadata = chunk.get("metadata", {})
                                    if isinstance(metadata, str):
                                        metadata = json.loads(metadata)

                                    if metadata.get("agent_should_terminate"):
                                        agent_should_terminate = True

                                        if content.get("function_name"):
                                            last_tool_call = content["function_name"]
                                        elif content.get("xml_tag_name"):
                                            last_tool_call = content["xml_tag_name"]

                                except Exception:
                                    pass

                            # 检查助手内容中的终止XML工具
                            if chunk.get("type") == "assistant" and "content" in chunk:
                                try:
                                    content = chunk.get("content", "{}")
                                    if isinstance(content, str):
                                        assistant_content_json = json.loads(content)
                                    else:
                                        assistant_content_json = content

                                    assistant_text = assistant_content_json.get(
                                        "content", ""
                                    )
                                    if isinstance(assistant_text, str):
                                        if "</ask>" in assistant_text:
                                            last_tool_call = "ask"
                                        elif "</complete>" in assistant_text:
                                            last_tool_call = "complete"

                                except (json.JSONDecodeError, Exception):
                                    pass

                            yield chunk
                    else:
                        # 非流式响应或错误字典
                        # logger.debug(f"响应不是异步可迭代的: {type(response)}")

                        # 检查是否是错误字典
                        if (
                            isinstance(response, dict)
                            and response.get("type") == "status"
                            and response.get("status") == "error"
                        ):
                            logger.error(
                                f"线程返回错误: {response.get('message', '未知错误')}"
                            )
                            error_detected = True
                            yield response
                        else:
                            logger.warning(f"意外的响应类型: {type(response)}")
                            error_detected = True

                    if error_detected:
                        break

                    if agent_should_terminate or last_tool_call in ["ask", "complete"]:
                        continue_execution = False

                except Exception as e:
                    # 使用ErrorProcessor进行安全错误处理
                    processed_error = ErrorProcessor.process_system_error(
                        e, context={"thread_id": self.config.thread_id}
                    )
                    ErrorProcessor.log_error(processed_error)
                    yield processed_error.to_stream_dict()
                    break

            except Exception as e:
                # 使用ErrorProcessor进行安全错误转换
                processed_error = ErrorProcessor.process_system_error(
                    e, context={"thread_id": self.config.thread_id}
                )
                ErrorProcessor.log_error(processed_error)
                yield processed_error.to_stream_dict()
                break


async def run_agent(
    thread_id: str,
    project_id: str,
    thread_manager: Optional[ThreadManager] = None,
    native_max_auto_continues: int = 25,
    max_iterations: int = 100,
    model_name: str = "glm-4.6",
    agent_config: Optional[dict] = None,
    cancellation_event: Optional[asyncio.Event] = None,
):
    effective_model = model_name

    config = AgentConfig(
        thread_id=thread_id,
        project_id=project_id,
        native_max_auto_continues=native_max_auto_continues,
        max_iterations=max_iterations,
        model_name=effective_model,
        agent_config=agent_config,
    )

    runner = AgentRunner(config)
    async for chunk in runner.run(cancellation_event=cancellation_event):
        yield chunk
