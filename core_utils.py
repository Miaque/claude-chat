import uuid
from typing import Optional

# 从专用模块导入并重新导出
from loguru import logger

from core.services import redis
from core.utils.run_management import cleanup_instance_runs

# 全局变量（将由 initialize 函数设置）
instance_id = None


async def cleanup():
    """清理资源并在关闭时停止正在运行的agent。"""
    logger.debug("开始清理 agent API 资源")

    # 清理实例特定的 agent 运行
    try:
        if instance_id:
            await cleanup_instance_runs(instance_id)
        else:
            logger.warning("实例 ID 未设置，无法清理实例特定的 agent 运行。")
    except Exception as e:
        logger.error(f"清理正在运行的 agent 失败: {str(e)}")

    # 关闭 Redis 连接
    await redis.close()
    logger.debug("已完成 agent API 资源清理")


def initialize(_instance_id: Optional[str] = None):
    """使用主 API 的资源初始化 agent API。"""
    global instance_id

    # 使用提供的 instance_id 或生成一个新的
    if _instance_id:
        instance_id = _instance_id
    else:
        # 生成实例 ID
        instance_id = str(uuid.uuid4())[:8]

    logger.debug(f"已使用实例 ID 初始化 agent API: {instance_id}")
