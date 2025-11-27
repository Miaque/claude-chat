"""Agent 运行管理工具 - 用于启动、停止和监控 agent 运行。"""

import json
from typing import Optional

from fastapi import HTTPException
from loguru import logger

from core.services import redis
from run_agent_background import _cleanup_redis_response_list, update_agent_run_status


async def cleanup_instance_runs(instance_id: str):
    """清理特定实例的所有运行中的 agent。"""
    logger.debug(f"开始清理实例 {instance_id} 的 agent 运行")

    try:
        if not instance_id:
            logger.warning("实例 ID 未设置，无法清理实例特定的 agent 运行。")
            return

        running_keys = await redis.keys(f"active_run:{instance_id}:*")
        logger.debug(
            f"找到 {len(running_keys)} 个需要清理的实例 {instance_id} 的运行中 agent"
        )

        for key in running_keys:
            # 键格式: active_run:{instance_id}:{agent_run_id}
            parts = key.split(":")
            if len(parts) == 3:
                agent_run_id = parts[2]
                await stop_agent_run_with_helpers(
                    agent_run_id, error_message=f"实例 {instance_id} 正在关闭"
                )
            else:
                logger.warning(f"发现意外的键格式: {key}")

    except Exception as e:
        logger.error(f"清理实例 {instance_id} 的运行中 agent 失败: {str(e)}")


async def stop_agent_run_with_helpers(
    agent_run_id: str, error_message: Optional[str] = None
):
    """
    停止 agent 运行并清理所有相关资源。

    此函数执行以下操作:
    1. 从 Redis 获取最终响应
    2. 更新数据库状态
    3. 向所有控制通道发布 STOP 信号
    4. 清理 Redis 键

    参数:
        agent_run_id: 要停止的 agent 运行的 ID
        error_message: 可选的错误消息（如果运行失败）
    """
    logger.debug(f"正在停止 agent 运行: {agent_run_id}")

    final_status = "failed" if error_message else "stopped"

    # 尝试从 Redis 获取最终响应
    response_list_key = f"agent_run:{agent_run_id}:responses"
    all_responses = []
    try:
        all_responses_json = await redis.lrange(response_list_key, 0, -1)
        all_responses = [json.loads(r) for r in all_responses_json]
        logger.debug(
            f"从Redis获取了{len(all_responses)}个响应，用于在停止/失败时更新数据库：{agent_run_id}"
        )
    except Exception as e:
        logger.error("在停止/失败期间，从Redis获取{}的响应失败：{}", agent_run_id, e)

    # 在数据库中更新 agent 运行状态
    update_success = await update_agent_run_status(
        agent_run_id, final_status, error=error_message
    )

    if not update_success:
        logger.error(f"更新已停止/失败的运行 {agent_run_id} 的数据库状态失败")
        raise HTTPException(status_code=500, detail="更新数据库中的agent运行状态失败")

    # 向全局控制通道发送 STOP 信号
    global_control_channel = f"agent_run:{agent_run_id}:control"
    try:
        await redis.publish(global_control_channel, "STOP")
        logger.debug(f"已向全局通道 {global_control_channel} 发布 STOP 信号")
    except Exception as e:
        logger.error("向全局通道 {} 发布 STOP 信号失败: {}", global_control_channel, e)

    # 查找处理此 agent 运行的所有实例，并向实例特定通道发送 STOP 信号
    try:
        instance_keys = await redis.keys(f"active_run:*:{agent_run_id}")
        logger.debug(
            f"为agent运行 {agent_run_id} 找到了 {len(instance_keys)} 个活跃实例keys"
        )

        for key in instance_keys:
            # 键格式: active_run:{instance_id}:{agent_run_id}
            parts = key.split(":")
            if len(parts) == 3:
                instance_id_from_key = parts[1]
                instance_control_channel = (
                    f"agent_run:{agent_run_id}:control:{instance_id_from_key}"
                )
                try:
                    await redis.publish(instance_control_channel, "STOP")
                    logger.debug(
                        f"已向实例通道 {instance_control_channel} 发布 STOP 信号"
                    )
                except Exception as e:
                    logger.warning(
                        f"向实例通道 {instance_control_channel} 发布 STOP 信号失败: {str(e)}"
                    )
            else:
                logger.warning(f"发现了不符合预期格式的key: {key}")

        # 在停止/失败时立即清理响应列表
        await _cleanup_redis_response_list(agent_run_id)

    except Exception as e:
        logger.error("无法找到或通知{}的活跃实例：{}", agent_run_id, e)

    logger.debug(f"已成功启动agent运行{agent_run_id}的停止流程")
