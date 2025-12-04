import json
import time
from typing import Any, Optional

from loguru import logger

from core.services import redis as redis_service

AGENT_CONFIG_TTL = 3600


def _get_cache_key(agent_id: str, version_id: Optional[str] = None) -> str:
    """获取 Agent 配置的缓存 Key"""
    if version_id:
        return f"agent_config:{agent_id}:{version_id}"
    return f"agent_config:{agent_id}:current"


def _get_user_mcps_key(agent_id: str) -> str:
    """获取用户专属 MCP 的缓存 Key"""
    return f"agent_mcps:{agent_id}"


async def get_cached_user_mcps(agent_id: str) -> Optional[dict[str, Any]]:
    """
    从缓存中读取用户专属 MCP
    返回包含 configured_mcps、custom_mcps、triggers 的字典
    """
    cache_key = _get_user_mcps_key(agent_id)

    try:
        cached = await redis_service.get(cache_key)
        if cached:
            data = json.loads(cached) if isinstance(cached, (str, bytes)) else cached
            logger.debug(f"用户专属 MCP 缓存命中：{agent_id}")
            return data
    except Exception as e:
        logger.warning(f"读取用户专属 MCP 缓存失败：{e}")

    return None


async def set_cached_user_mcps(agent_id: str, configured_mcps: list) -> None:
    """缓存用户专属的 MCP"""
    cache_key = _get_user_mcps_key(agent_id)
    data = {"configured_mcps": configured_mcps}

    try:
        await redis_service.set(cache_key, json.dumps(data), ex=AGENT_CONFIG_TTL)
        logger.debug(f"缓存用户专属的 MCP: {agent_id}")
    except Exception as e:
        logger.warning(f"缓存用户专属的 MCP 失败：{e}")


async def get_cached_agent_config(agent_id: str, version_id: Optional[str] = None) -> Optional[dict[str, Any]]:
    """
    从缓存读取 Agent 配置

    仅适用于自定义 Agent - 使用 get_static_suna_config() + get_cached_user_mcps()。
    """
    cache_key = _get_cache_key(agent_id, version_id)

    try:
        cached = await redis_service.get(cache_key)
        if cached:
            data = json.loads(cached) if isinstance(cached, (str, bytes)) else cached
            logger.debug(f"Agent 配置缓存命中: {agent_id}")
            return data
    except Exception as e:
        logger.warning(f"读取 Agent 配置缓存失败：{e}")

    return None


async def set_cached_agent_config(
    agent_id: str, config: dict[str, Any], version_id: Optional[str] = None, is_global_default: bool = False
) -> None:
    """缓存完整的 Agent 配置"""
    if is_global_default:
        await set_cached_user_mcps(agent_id, config.get("configured_mcps", []))
        return

    cache_key = _get_cache_key(agent_id, version_id)

    try:
        await redis_service.set(cache_key, json.dumps(config), ex=AGENT_CONFIG_TTL)
        logger.debug(f"缓存 Agent 配置: {agent_id}")
    except Exception as e:
        logger.warning(f"缓存 Agent 配置失败：{e}")


async def invalidate_agent_config_cache(agent_id: str) -> None:
    """清除 Agent 配置缓存"""
    try:
        await redis_service.delete(f"agent_config:{agent_id}:current")
        await redis_service.delete(f"agent_mcps:{agent_id}")
        logger.info(f"清除 Agent 配置缓存: {agent_id}")
    except Exception as e:
        logger.warning(f"清除 Agent 配置缓存失败：{e}")


PROJECT_CACHE_TTL = 300


def _get_project_cache_key(project_id: str) -> str:
    """获取项目元数据的缓存 Key"""
    return f"project_meta:{project_id}"


async def get_cached_project_metadata(project_id: str) -> Optional[dict[str, Any]]:
    cache_key = _get_project_cache_key(project_id)

    try:
        cached = await redis_service.get(cache_key)
        if cached:
            data = json.loads(cached) if isinstance(cached, (str, bytes)) else cached
            logger.debug(f"项目元数据缓存命中：{project_id}")
            return data
    except Exception as e:
        logger.warning(f"读取项目元数据缓存失败：{e}")

    return None


async def set_cached_project_metadata(project_id: str, sandbox: dict[str, Any]) -> None:
    """缓存项目元数据"""
    cache_key = _get_project_cache_key(project_id)
    data = {"project_id": project_id, "sandbox": sandbox}

    try:
        await redis_service.set(cache_key, json.dumps(data), ex=PROJECT_CACHE_TTL)
        logger.debug(f"缓存项目元数据: {project_id}")
    except Exception as e:
        logger.warning(f"缓存项目元数据失败：{e}")


async def invalidate_project_cache(project_id: str) -> None:
    """清除项目元数据缓存"""
    try:
        await redis_service.delete(_get_project_cache_key(project_id))
        logger.debug(f"清除项目元数据缓存: {project_id}")
    except Exception as e:
        logger.warning(f"清除项目元数据缓存失败：{e}")


RUNNING_RUNS_TTL = 5


def _get_running_runs_key(account_id: str) -> str:
    return f"running_runs:{account_id}"


async def get_cached_running_runs(account_id: str) -> Optional[dict[str, Any]]:
    cache_key = _get_running_runs_key(account_id)

    try:
        cached = await redis_service.get(cache_key)
        if cached:
            data = json.loads(cached) if isinstance(cached, (str, bytes)) else cached
            logger.debug(f"运行中数量缓存命中: {account_id}")
            return data
    except Exception as e:
        logger.warning(f"读取运行中数量缓存失败：{e}")

    return None


async def set_cached_running_runs(account_id: str, running_count: int, running_thread_ids: list) -> None:
    """缓存运行中数量"""
    cache_key = _get_running_runs_key(account_id)
    data = {"running_count": running_count, "running_thread_ids": running_thread_ids, "cached_at": time.time()}

    try:
        await redis_service.set(cache_key, json.dumps(data), ex=RUNNING_RUNS_TTL)
        logger.debug(f"缓存运行中数量: {account_id} ({running_count} 个)")
    except Exception as e:
        logger.warning(f"缓存运行中数量失败：{e}")


async def invalidate_running_runs_cache(account_id: str) -> None:
    """清除运行中数量缓存"""
    try:
        await redis_service.delete(_get_running_runs_key(account_id))
        logger.debug(f"清除运行中数量缓存: {account_id}")
    except Exception as e:
        logger.warning(f"清除运行中数量缓存失败：{e}")


THREAD_COUNT_TTL = 300


def _get_thread_count_key(account_id: str) -> str:
    """获取线程数缓存的 Key"""
    return f"thread_count:{account_id}"


async def get_cached_thread_count(account_id: str) -> Optional[int]:
    """从缓存中读取线程数"""
    cache_key = _get_thread_count_key(account_id)

    try:
        cached = await redis_service.get(cache_key)
        if cached is not None:
            count = int(cached) if isinstance(cached, (str, bytes)) else cached
            logger.debug(f"线程数缓存命中：{account_id}（{count} 个）")
            return count
    except Exception as e:
        logger.warning(f"读取线程数缓存失败：{e}")

    return None


async def set_cached_thread_count(account_id: str, count: int) -> None:
    """缓存线程数"""
    cache_key = _get_thread_count_key(account_id)

    try:
        await redis_service.set(cache_key, str(count), ex=THREAD_COUNT_TTL)
        logger.debug(f"缓存线程数: {account_id} ({count} 个)")
    except Exception as e:
        logger.warning(f"缓存线程数失败：{e}")


async def increment_thread_count_cache(account_id: str) -> None:
    """新建线程时，缓存里的线程数加 1"""
    cache_key = _get_thread_count_key(account_id)

    try:
        current = await redis_service.get(cache_key)
        if current is not None:
            await redis_service.incr(cache_key)
            logger.debug(f"线程数 +1: {account_id}")
    except Exception as e:
        logger.warning(f"线程数 +1 失败：{e}")


async def invalidate_thread_count_cache(account_id: str) -> None:
    """删除线程时，清除线程数缓存"""
    try:
        await redis_service.delete(_get_thread_count_key(account_id))
        logger.debug(f"清除线程数缓存: {account_id}")
    except Exception as e:
        logger.warning(f"清除线程数缓存失败：{e}")
