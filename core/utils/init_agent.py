from loguru import logger

from core.utils.default_agent_service import DefaultAgentService
from models.agent import Agents

_installation_cache = set[str]()
_installation_in_progress = set[str]()


async def ensure_agent_initialized(account_id: str) -> None:
    if account_id in _installation_cache:
        return

    if account_id in _installation_in_progress:
        return

    try:
        _installation_in_progress.add(account_id)

        agent_id = Agents.get_global_default_agent_id(account_id)

        if agent_id:
            _installation_cache.add(account_id)
            return

        service = DefaultAgentService()
        agent_id = await service.init_agent_for_user(account_id, replace_existing=False)

        if agent_id:
            _installation_cache.add(account_id)
    except Exception as e:
        logger.error("为用户 {} 初始化默认智能体失败: {e}", account_id, e)
    finally:
        _installation_in_progress.discard(account_id)
