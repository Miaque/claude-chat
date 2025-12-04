import asyncio
from typing import Any, Optional

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
    redis_db = app_config.REDIS_DB

    # 连接池配置 - 为生产环境优化
    max_connections = 128  # 生产环境的合理限制
    socket_timeout = 15.0  # 15 秒套接字超时
    connect_timeout = 10.0  # 10 秒连接超时
    retry_on_timeout = app_config.REDIS_RETRY_ON_TIMEOUT

    logger.info(f"初始化 Redis 连接池到 {redis_host}:{redis_port}，最大连接数 {max_connections}")

    # 使用生产环境优化设置创建连接池
    pool = redis.ConnectionPool(
        host=redis_host,
        port=redis_port,
        db=redis_db,
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
        except TimeoutError:
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
        except TimeoutError:
            logger.warning("Redis 关闭超时，强制关闭")
        except Exception as e:
            logger.warning(f"关闭 Redis 客户端时出错: {e}")
        finally:
            client = None

    if pool:
        # logger.debug("正在关闭 Redis 连接池")
        try:
            await asyncio.wait_for(pool.aclose(), timeout=5.0)
        except TimeoutError:
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


async def lrange(key: str, start: int, end: int) -> list[str]:
    """从列表获取指定范围的元素。"""
    redis_client = await get_client()
    return await redis_client.lrange(key, start, end)


# 键管理


async def keys(pattern: str) -> list[str]:
    """根据模式获取匹配的键。"""
    redis_client = await get_client()
    return await redis_client.keys(pattern)


async def expire(key: str, seconds: int):
    """为键设置过期时间。"""
    redis_client = await get_client()
    return await redis_client.expire(key, seconds)


# Stream 操作


async def xadd(
    stream_key: str,
    fields: dict[str, Any],
    maxlen: Optional[int] = None,
    approximate: bool = True,
) -> str:
    """
    向 Stream 添加消息。

    参数:
        stream_key: Stream 的键名
        fields: 消息字段字典
        maxlen: 可选的最大长度限制
        approximate: 是否使用近似修剪（默认 True，性能更好）

    返回:
        消息 ID
    """
    redis_client = await get_client()
    return await redis_client.xadd(stream_key, fields, maxlen=maxlen, approximate=approximate)


async def xread(
    streams: dict[str, str],
    count: Optional[int] = None,
    block: Optional[int] = None,
) -> list | None:
    """
    从一个或多个 Stream 读取消息。

    参数:
        streams: 字典，键为 stream 名称，值为起始消息 ID（使用 "0" 从头读取，"$" 读取新消息）
        count: 每个 stream 最多返回的消息数量
        block: 阻塞等待的毫秒数（None 表示不阻塞）

    返回:
        消息列表，格式为 [[stream_name, [(msg_id, {fields}), ...]], ...]
        如果没有消息则返回 None
    """
    redis_client = await get_client()
    return await redis_client.xread(streams, count=count, block=block)


async def xrange(
    stream_key: str,
    start: str = "-",
    end: str = "+",
    count: Optional[int] = None,
) -> list:
    """
    获取 Stream 中指定范围的消息。

    参数:
        stream_key: Stream 的键名
        start: 起始消息 ID（"-" 表示最早的消息）
        end: 结束消息 ID（"+" 表示最新的消息）
        count: 最多返回的消息数量

    返回:
        消息列表，格式为 [(msg_id, {fields}), ...]
    """
    redis_client = await get_client()
    return await redis_client.xrange(stream_key, start, end, count=count)


async def incr(key: str):
    redis_client = await get_client()
    return await redis_client.incr(key)


async def decr(key: str):
    redis_client = await get_client()
    return await redis_client.decr(key)
