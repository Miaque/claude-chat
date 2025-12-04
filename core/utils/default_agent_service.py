from datetime import datetime
from typing import Optional

from loguru import logger

from models.agent import Agent, Agents


class DefaultAgentService:
    def __init__(self):
        pass

    async def init_agent_for_user(self, account_id: str, replace_existing: bool = False) -> Optional[str]:
        """初始化用户默认智能体"""
        logger.debug(f"正在为用户 {account_id} 初始化默认智能体")

        try:
            existing_agent_id = Agents.get_global_default_agent_id(account_id)
            if existing_agent_id:
                return existing_agent_id

            agent_id = await self._create_agent_for_user(account_id)
            logger.debug(f"成功为用户 {account_id} 初始化默认智能体 {agent_id}")
            return agent_id

        except Exception as e:
            logger.error("初始化用户默认智能体失败: {}", e)
            return None

    async def _create_agent_for_user(self, account_id: str) -> str:
        """为用户创建默认智能体"""

        agent_data = {
            "account_id": account_id,
            "name": "default",
            "description": "默认智能体",
            "is_default": True,
            "icon_name": "sun",
            "icon_color": "#FFFFFF",
            "icon_background": "#000000",
            "meta": {
                "is_global_default": True,
                "installation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "version_count": 1,
        }

        agent = Agents.insert(Agent(**agent_data))
        if not agent:
            raise Exception("创建默认智能体失败")

        return agent.agent_id
