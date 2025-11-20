import asyncio
from typing import Any, List, Optional

import redis.asyncio as redis
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from configs import app_config

# Redis 客户端和连接池
client: redis.Redis | None = None
pool: redis.ConnectionPool | None = None
_initialized = False
_init_lock = asyncio.Lock()

# 常量
REDIS_KEY_TTL = 3600 * 24  # 24 小时 TTL 作为安全机制


def initialize():
    """使用环境变量初始化 Redis 连接池和客户端。"""
    global client, pool

    # 获取 Redis 配置
    redis_host = app_config.REDIS_HOST
    redis_port = app_config.REDIS_PORT
    redis_password = app_config.REDIS_PASSWORD

    # 连接池配置 - 为生产环境优化
    max_connections = 128  # 生产环境的合理限制
    socket_timeout = 15.0  # 15 秒套接字超时
    connect_timeout = 10.0  # 10 秒连接超时
    retry_on_timeout = app_config.REDIS_RETRY_ON_TIMEOUT

    logger.info(
        f"初始化 Redis 连接池到 {redis_host}:{redis_port}，最大连接数 {max_connections}"
    )

    # 使用生产环境优化设置创建连接池
    pool = redis.ConnectionPool(
        host=redis_host,
        port=redis_port,
        password=redis_password,
        decode_responses=True,
        socket_timeout=socket_timeout,
        socket_connect_timeout=connect_timeout,
        socket_keepalive=True,
        retry_on_timeout=retry_on_timeout,
        health_check_interval=30,
        max_connections=max_connections,
    )

    # 从连接池创建 Redis 客户端
    client = redis.Redis(connection_pool=pool)

    return client


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def initialize_async():
    """异步初始化 Redis 连接。"""
    global client, _initialized

    async with _init_lock:
        if not _initialized:
            # logger.debug("正在初始化 Redis 连接")
            initialize()

        try:
            # 带超时测试连接
            await asyncio.wait_for(client.ping(), timeout=5.0)
            logger.info("成功连接到 Redis")
            _initialized = True
        except asyncio.TimeoutError:
            logger.error("初始化期间 Redis 连接超时")
            client = None
            _initialized = False
            raise ConnectionError("Redis 连接超时")
        except Exception as e:
            logger.error(f"连接 Redis 失败: {e}")
            client = None
            _initialized = False
            raise

    return client


async def close():
    """关闭 Redis 连接和连接池。"""
    global client, pool, _initialized
    if client:
        # logger.debug("正在关闭 Redis 连接")
        try:
            await asyncio.wait_for(client.aclose(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Redis 关闭超时，强制关闭")
        except Exception as e:
            logger.warning(f"关闭 Redis 客户端时出错: {e}")
        finally:
            client = None

    if pool:
        # logger.debug("正在关闭 Redis 连接池")
        try:
            await asyncio.wait_for(pool.aclose(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Redis 连接池关闭超时，强制关闭")
        except Exception as e:
            logger.warning(f"关闭 Redis 连接池时出错: {e}")
        finally:
            pool = None

    _initialized = False
    logger.info("Redis 连接和连接池已关闭")


async def get_client():
    """获取 Redis 客户端，如有必要则初始化。"""
    global client, _initialized
    if client is None or not _initialized:
        await initialize_async()
    return client


# 基本 Redis 操作
async def set(key: str, value: str, ex: Optional[int] = None, nx: bool = False):
    """设置 Redis 键。"""
    redis_client = await get_client()
    return await redis_client.set(key, value, ex=ex, nx=nx)


async def get(key: str, default: Optional[str] = None):
    """获取 Redis 键。"""
    redis_client = await get_client()
    result = await redis_client.get(key)
    return result if result is not None else default


async def delete(key: str):
    """删除 Redis 键。"""
    redis_client = await get_client()
    return await redis_client.delete(key)


async def publish(channel: str, message: str):
    """向 Redis 频道发布消息。"""
    redis_client = await get_client()
    return await redis_client.publish(channel, message)


async def create_pubsub():
    """创建 Redis pubsub 对象。"""
    redis_client = await get_client()
    return redis_client.pubsub()


# 列表操作
async def rpush(key: str, *values: Any):
    """将一个或多个值追加到列表。"""
    redis_client = await get_client()
    return await redis_client.rpush(key, *values)


async def lrange(key: str, start: int, end: int) -> List[str]:
    """从列表获取指定范围的元素。"""
    redis_client = await get_client()
    return await redis_client.lrange(key, start, end)


# 键管理


async def keys(pattern: str) -> List[str]:
    """根据模式获取匹配的键。"""
    redis_client = await get_client()
    return await redis_client.keys(pattern)


async def expire(key: str, seconds: int):
    """为键设置过期时间。"""
    redis_client = await get_client()
    return await redis_client.expire(key, seconds)
