import asyncio
import datetime
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Optional

from claude_agent_sdk.types import PermissionMode
from loguru import logger

from core.error_processor import ErrorProcessor
from core.prompts.prompt import get_system_prompt
from core.response_processor import ProcessorConfig
from core.thread_manager import ThreadManager
from models.message import Messages
from models.project import Projects
from models.thread import Thread, Threads


@dataclass
class AgentConfig:
    thread_id: str
    project_id: str
    model_name: str = "glm-4.6"
    agent_config: Optional[dict] = None
    account_id: Optional[str] = None
    permission_mode: PermissionMode | None = None


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

        now = datetime.datetime.now()
        datetime_info = "\n\n=== 当前日期/时间信息 ===\n"
        datetime_info += f"今天的日期: {now.strftime('%A, %B %d, %Y')}\n"
        datetime_info += f"当前年份: {now.strftime('%Y')}\n"
        datetime_info += f"当前月份: {now.strftime('%B')}\n"
        datetime_info += f"当前日期: {now.strftime('%A')}\n"
        datetime_info += "将此信息用于任何时间敏感的任务、研究，或需要当前日期/时间上下文时。\n"

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
                logger.debug(f"为用户 {user_id} 添加了地区上下文 ({locale}) 到系统提示中")
            except Exception as e:
                logger.warning(f"向系统提示添加地区上下文失败: {e}")

        system_message = {"role": "system", "content": system_content}
        return system_message


class AgentRunner:
    def __init__(self, config: AgentConfig):
        self.config = config

    async def setup(self):
        self.thread_manager = ThreadManager(agent_config=self.config.agent_config)

        response = Threads.get_by_id(self.config.thread_id, Thread.account_id)
        if not response:
            raise ValueError(f"未找到线程 {self.config.thread_id}")

        self.account_id = response.account_id

        if not self.account_id:
            raise ValueError(f"线程 {self.config.thread_id} 没有关联的账户")

        project_data = Projects.get_by_id(self.config.project_id)
        if not project_data:
            raise ValueError(f"未找到项目 {self.config.project_id}")

        sandbox_info = project_data.sandbox
        if not sandbox_info.get("id"):
            logger.debug(f"未找到项目 {self.config.project_id} 的sandbox；将在需要时延迟创建")

    async def run(self, cancellation_event: asyncio.Event | None) -> AsyncGenerator[dict[str, Any], None]:
        await self.setup()

        system_message = await PromptManager.build_system_prompt(
            self.config.model_name,
            self.config.agent_config,
            self.config.thread_id,
            # tool_registry=self.thread_manager.tool_registry,
            user_id=self.account_id,
        )
        logger.info(f"系统消息构建完成: {len(str(system_message.get('content', '')))} 字符")
        logger.debug(f"收到 model_name: {self.config.model_name}")

        latest_user_message = Messages.get_latest_user_message(self.config.thread_id)

        latest_user_message_content = None
        if latest_user_message:
            data = latest_user_message.content
            if isinstance(data, str):
                data = json.loads(data)
            # 提取内容用于快速路径优化
            latest_user_message_content = data.get("content") if isinstance(data, dict) else str(data)

        temporary_message = None
        try:
            logger.debug(f"开始为 {self.config.thread_id} 执行线程")
            response = await self.thread_manager.run_thread(
                thread_id=self.config.thread_id,
                system_prompt=system_message,
                stream=True,
                permission_mode=self.config.permission_mode,
                llm_model=self.config.model_name,
                tool_choice="auto",
                temporary_message=temporary_message,
                latest_user_message_content=latest_user_message_content,
                processor_config=ProcessorConfig(
                    execute_on_stream=True,
                ),
                cancellation_event=cancellation_event,
            )

            try:
                if hasattr(response, "__aiter__") and not isinstance(response, dict):
                    async for chunk in response:
                        # 检查来自thread_manager的错误状态
                        if isinstance(chunk, dict) and chunk.get("type") == "status" and chunk.get("status") == "error":
                            logger.error(f"线程执行出错: {chunk.get('message', '未知错误')}")
                            yield chunk
                            continue

                        # 检查流中的错误状态（消息格式）
                        if isinstance(chunk, dict) and chunk.get("type") == "status":
                            try:
                                content = chunk.get("content", {})
                                if isinstance(content, str):
                                    content = json.loads(content)

                                # 检查错误状态
                                if content.get("status_type") == "error":
                                    yield chunk
                                    continue

                            except Exception:
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
                        logger.error(f"线程返回错误: {response.get('message', '未知错误')}")
                        yield response
                    else:
                        logger.warning(f"意外的响应类型: {type(response)}")

            except Exception as e:
                # 使用ErrorProcessor进行安全错误处理
                processed_error = ErrorProcessor.process_system_error(e, context={"thread_id": self.config.thread_id})
                ErrorProcessor.log_error(processed_error)
                yield processed_error.to_stream_dict()

        except Exception as e:
            # 使用ErrorProcessor进行安全错误转换
            processed_error = ErrorProcessor.process_system_error(e, context={"thread_id": self.config.thread_id})
            ErrorProcessor.log_error(processed_error)
            yield processed_error.to_stream_dict()


async def run_agent(
    thread_id: str,
    project_id: str,
    thread_manager: Optional[ThreadManager] = None,
    model_name: str = "glm-4.6",
    agent_config: Optional[dict] = None,
    cancellation_event: Optional[asyncio.Event] = None,
    account_id: Optional[str] = None,
    permission_mode: PermissionMode | None = None,
):
    config = AgentConfig(
        thread_id=thread_id,
        project_id=project_id,
        model_name=model_name,
        agent_config=agent_config,
        account_id=account_id,
        permission_mode=permission_mode,
    )

    runner = AgentRunner(config)
    async for chunk in runner.run(cancellation_event=cancellation_event):
        yield chunk
