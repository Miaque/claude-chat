import asyncio
import json
import traceback
import uuid
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import dramatiq
import structlog
from dramatiq.brokers.redis import RedisBroker
from loguru import logger
from tenacity import AsyncRetrying, stop_after_attempt, wait_fixed

from configs import app_config
from core.run import run_agent
from core.services import redis

logger.info(
    f"正在配置 Dramatiq 代理，Redis 地址: {app_config.REDIS_HOST}:{app_config.REDIS_PORT}"
)
redis_broker = RedisBroker(
    host=app_config.REDIS_HOST,
    port=app_config.REDIS_PORT,
    password=app_config.REDIS_PASSWORD,
    middleware=[dramatiq.middleware.AsyncIO()],
)

dramatiq.set_broker(redis_broker)

_initialized = False
instance_id = ""


async def initialize():
    """使用主 API 的资源初始化 Agent API。"""
    global instance_id, _initialized

    if _initialized:
        return  # 已经初始化

    if not instance_id:
        instance_id = str(uuid.uuid4())[:8]

    logger.info(
        f"正在初始化工作进程，Redis 地址: {app_config.REDIS_HOST}:{app_config.REDIS_PORT}"
    )
    await redis.initialize_async()

    _initialized = True
    logger.info(f"✅ 工作进程初始化成功，实例 ID: {instance_id}")


@dramatiq.actor
async def check_health(key: str):
    """使用 Redis 在后台运行 Agent。"""
    structlog.contextvars.clear_contextvars()
    await redis.set(key, "healthy", ex=redis.REDIS_KEY_TTL)


@dramatiq.actor
async def run_agent_background(
    agent_run_id: str,
    thread_id: str,
    instance_id: str,
    project_id: str,
    model_name: str = "glm-4.6",
    agent_config: Optional[dict] = None,
    request_id: Optional[str] = None,
):
    """使用 Redis 在后台运行 Agent。"""
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        agent_run_id=agent_run_id,
        thread_id=thread_id,
        request_id=request_id,
    )

    try:
        await initialize()
    except Exception as e:
        logger.critical(f"初始化 Redis 连接失败: {e}")
        raise e

    # 幂等性检查：防止重复运行
    run_lock_key = f"agent_run_lock:{agent_run_id}"

    # 尝试获取此 Agent 运行的锁
    lock_acquired = await redis.set(
        run_lock_key, instance_id, nx=True, ex=redis.REDIS_KEY_TTL
    )

    if not lock_acquired:
        # 检查是否已在其他实例中处理
        existing_instance = await redis.get(run_lock_key)
        if existing_instance:
            logger.info(
                f"Agent 运行 {agent_run_id} 已在实例 {existing_instance.decode() if isinstance(existing_instance, bytes) else existing_instance} 中处理，跳过重复执行。"
            )
            return
        else:
            # 锁存在但无值，尝试重新获取
            lock_acquired = await redis.set(
                run_lock_key, instance_id, nx=True, ex=redis.REDIS_KEY_TTL
            )
            if not lock_acquired:
                logger.info(
                    f"Agent 运行 {agent_run_id} 已在其他实例中处理，跳过重复执行。"
                )
                return

    logger.info(
        f"开始后台 Agent 运行: {agent_run_id}，线程: {thread_id} (实例: {instance_id})"
    )

    client = await db.client
    start_time = datetime.now(ZoneInfo("Asia/Shanghai"))
    total_responses = 0
    pubsub = None
    stop_checker = None
    stop_signal_received = False

    # 创建取消事件以通知 LLM 停止
    cancellation_event = asyncio.Event()

    # 定义 Redis 键和通道
    response_list_key = f"agent_run:{agent_run_id}:responses"
    response_channel = f"agent_run:{agent_run_id}:new_response"
    instance_control_channel = f"agent_run:{agent_run_id}:control:{instance_id}"
    global_control_channel = f"agent_run:{agent_run_id}:control"
    instance_active_key = f"active_run:{instance_id}:{agent_run_id}"

    async def check_for_stop_signal():
        nonlocal stop_signal_received
        if not pubsub:
            return
        try:
            while not stop_signal_received:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=0.5
                )
                if message and message.get("type") == "message":
                    data = message.get("data")
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")
                    if data == "STOP":
                        logger.debug(
                            f"接收到 STOP 信号，Agent 运行: {agent_run_id} (实例: {instance_id})"
                        )
                        stop_signal_received = True
                        # 设置取消事件以立即停止 LLM 执行
                        cancellation_event.set()
                        break
                # 定期刷新活动运行键的 TTL
                if total_responses % 50 == 0:  # 每约 50 个响应刷新一次
                    try:
                        await redis.expire(instance_active_key, redis.REDIS_KEY_TTL)
                    except Exception as ttl_err:
                        logger.warning(
                            f"刷新 TTL 失败: {instance_active_key}: {ttl_err}"
                        )
                await asyncio.sleep(0.1)  # 短暂休眠防止循环过紧
        except asyncio.CancelledError:
            logger.debug(f"停止信号检查器已取消: {agent_run_id} (实例: {instance_id})")
        except Exception as e:
            logger.error(f"停止信号检查器出错: {agent_run_id}: {e}", exc_info=True)
            stop_signal_received = True  # 检查器失败时停止运行

    try:
        # 设置 Pub/Sub 监听器以接收控制信号
        pubsub = await redis.create_pubsub()
        try:
            retry = AsyncRetrying(stop=stop_after_attempt(3), wait=wait_fixed(1))
            async for attempt in retry:
                with attempt:
                    await pubsub.subscribe(
                        instance_control_channel, global_control_channel
                    )
        except Exception as e:
            logger.error(f"Redis 订阅控制通道失败: {e}", exc_info=True)
            raise e

        logger.info(
            f"已订阅控制通道: {instance_control_channel}, {global_control_channel}"
        )
        stop_checker = asyncio.create_task(check_for_stop_signal())

        # 确保活动运行键存在并设置 TTL
        await redis.set(instance_active_key, "running", ex=redis.REDIS_KEY_TTL)

        # 使用取消事件初始化 Agent 生成器
        agent_gen = run_agent(
            thread_id=thread_id,
            project_id=project_id,
            model_name=effective_model,
            agent_config=agent_config,
            cancellation_event=cancellation_event,
        )

        final_status = "running"
        error_message = None

        pending_redis_operations = []

        async for response in agent_gen:
            if stop_signal_received:
                logger.debug(f"Agent 运行 {agent_run_id} 已被信号停止。")
                final_status = "stopped"
                break

            # 将响应存储到 Redis 列表并发布通知
            response_json = json.dumps(response)
            pending_redis_operations.append(
                asyncio.create_task(redis.rpush(response_list_key, response_json))
            )
            pending_redis_operations.append(
                asyncio.create_task(redis.publish(response_channel, "new"))
            )
            total_responses += 1

            # 检查 Agent 是否发出完成或错误信号
            if response.get("type") == "status":
                status_val = response.get("status")
                # logger.debug(f"Agent status: {status_val}")

                if status_val in ["completed", "failed", "stopped", "error"]:
                    logger.info(f"Agent 运行 {agent_run_id} 完成，状态: {status_val}")
                    final_status = status_val if status_val != "error" else "failed"
                    if status_val in ["failed", "stopped", "error"]:
                        error_message = response.get(
                            "message", f"Run ended with status: {status_val}"
                        )
                        logger.error(f"Agent 运行失败: {error_message}")
                    break

        # 如果循环结束但没有明确的完成/错误/停止信号，则标记为已完成
        if final_status == "running":
            final_status = "completed"
            duration = (
                datetime.now(ZoneInfo("Asia/Shanghai")) - start_time
            ).total_seconds()
            logger.info(
                f"Agent 运行 {agent_run_id} 正常完成 (持续: {duration:.2f}秒, 响应数: {total_responses})"
            )
            completion_message = {
                "type": "status",
                "status": "completed",
                "message": "Agent run completed successfully",
            }
            await redis.rpush(response_list_key, json.dumps(completion_message))
            await redis.publish(response_channel, "new")  # 通知完成消息

        # 从 Redis 获取最终响应以更新数据库
        all_responses_json = await redis.lrange(response_list_key, 0, -1)
        all_responses = [json.loads(r) for r in all_responses_json]

        # 更新数据库状态
        await update_agent_run_status(
            client, agent_run_id, final_status, error=error_message
        )

        # 发布最终控制信号 (END_STREAM 或 ERROR)
        control_signal = (
            "END_STREAM"
            if final_status == "completed"
            else "ERROR"
            if final_status == "failed"
            else "STOP"
        )
        try:
            await redis.publish(global_control_channel, control_signal)
            # 无需发布到实例通道，因为运行正在此实例上结束
            logger.debug(
                f"已发布最终控制信号 '{control_signal}' 到 {global_control_channel}"
            )
        except Exception as e:
            logger.warning(f"发布最终控制信号失败 {control_signal}: {str(e)}")

    except Exception as e:
        error_message = str(e)
        traceback_str = traceback.format_exc()
        duration = (
            datetime.now(ZoneInfo("Asia/Shanghai")) - start_time
        ).total_seconds()
        logger.error(
            f"Agent 运行 {agent_run_id} 运行 {duration:.2f}秒后出错: {error_message}\n{traceback_str} (实例: {instance_id})"
        )
        final_status = "failed"

        # 将错误消息推送到 Redis 列表
        error_response = {"type": "status", "status": "error", "message": error_message}
        try:
            await redis.rpush(response_list_key, json.dumps(error_response))
            await redis.publish(response_channel, "new")
        except Exception as redis_err:
            logger.error(f"将错误响应推送到 Redis 失败: {agent_run_id}: {redis_err}")

        # 更新数据库状态
        await update_agent_run_status(
            client, agent_run_id, "failed", error=f"{error_message}\n{traceback_str}"
        )

        # 发布 ERROR 信号
        try:
            await redis.publish(global_control_channel, "ERROR")
            logger.debug(f"发布 ERROR 信号到 {global_control_channel}")
        except Exception as e:
            logger.warning(f"发布 ERROR 信号失败: {str(e)}")

    finally:
        # 清理停止检查器任务
        if stop_checker and not stop_checker.done():
            stop_checker.cancel()
            try:
                await stop_checker
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"停止检查器取消时出错: {e}")

        # 关闭 pubsub 连接
        if pubsub:
            try:
                await pubsub.unsubscribe()
                await pubsub.close()
                logger.debug(f"已关闭 {agent_run_id} 的 pubsub 连接")
            except Exception as e:
                logger.warning(f"关闭 {agent_run_id} 的 pubsub 时出错: {str(e)}")

        # 在 Redis 中设置响应列表的 TTL
        await _cleanup_redis_response_list(agent_run_id)

        # 移除实例特定的活动运行键
        await _cleanup_redis_instance_key(agent_run_id)

        # 清理运行锁
        await _cleanup_redis_run_lock(agent_run_id)

        # 等待所有待处理的 Redis 操作完成（带超时）
        try:
            await asyncio.wait_for(
                asyncio.gather(*pending_redis_operations), timeout=30.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"等待 {agent_run_id} 的 Redis 操作超时")

        logger.debug(
            f"Agent 后台运行任务已全部完成: {agent_run_id} (实例: {instance_id})，最终状态: {final_status}"
        )


async def _cleanup_redis_instance_key(agent_run_id: str):
    """清理 Agent 运行的实例特定 Redis 键。"""
    if not instance_id:
        logger.warning("实例 ID 未设置，无法清理实例键。")
        return
    key = f"active_run:{instance_id}:{agent_run_id}"
    # logger.debug(f"正在清理 Redis 实例键: {key}")
    try:
        await redis.delete(key)
        # logger.debug(f"成功清理 Redis 键: {key}")
    except Exception as e:
        logger.warning(f"清理 Redis 键失败 {key}: {str(e)}")


async def _cleanup_redis_run_lock(agent_run_id: str):
    """清理 Agent 运行的运行锁 Redis 键。"""
    run_lock_key = f"agent_run_lock:{agent_run_id}"
    # logger.debug(f"正在清理 Redis 运行锁键: {run_lock_key}")
    try:
        await redis.delete(run_lock_key)
        # logger.debug(f"成功清理 Redis 运行锁键: {run_lock_key}")
    except Exception as e:
        logger.warning(f"清理 Redis 运行锁键失败 {run_lock_key}: {str(e)}")


# Redis 响应列表的 TTL（24 小时）
REDIS_RESPONSE_LIST_TTL = 3600 * 24


async def _cleanup_redis_response_list(agent_run_id: str):
    """在 Redis 响应列表上设置 TTL。"""
    response_list_key = f"agent_run:{agent_run_id}:responses"
    try:
        await redis.expire(response_list_key, REDIS_RESPONSE_LIST_TTL)
        # logger.debug(f"已设置响应列表 TTL ({REDIS_RESPONSE_LIST_TTL}秒): {response_list_key}")
    except Exception as e:
        logger.warning(f"设置响应列表 TTL 失败 {response_list_key}: {str(e)}")


async def update_agent_run_status(
    client,
    agent_run_id: str,
    status: str,
    error: Optional[str] = None,
) -> bool:
    """
    Centralized function to update agent run status.
    Returns True if update was successful.
    """
    try:
        update_data = {
            "status": status,
            "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        }

        if error:
            update_data["error"] = error

        # 最多重试 3 次
        for retry in range(3):
            try:
                update_result = (
                    await client.table("agent_runs")
                    .update(update_data)
                    .eq("id", agent_run_id)
                    .execute()
                )

                if hasattr(update_result, "data") and update_result.data:
                    # logger.debug(f"成功更新 Agent 运行 {agent_run_id} 状态为 '{status}' (重试 {retry})")

                    # 验证更新
                    verify_result = (
                        await client.table("agent_runs")
                        .select("status", "completed_at")
                        .eq("id", agent_run_id)
                        .execute()
                    )
                    if verify_result.data:
                        actual_status = verify_result.data[0].get("status")
                        completed_at = verify_result.data[0].get("completed_at")
                        # logger.debug(f"验证 Agent 运行更新: status={actual_status}, completed_at={completed_at}")
                    return True
                else:
                    logger.warning(
                        f"数据库更新未返回数据，Agent 运行: {agent_run_id}，重试: {retry}: {update_result}"
                    )
                    if retry == 2:  # 最后一次重试
                        logger.error(
                            f"所有重试后更新 Agent 运行状态失败: {agent_run_id}"
                        )
                        return False
            except Exception as db_error:
                logger.error(
                    f"更新状态时数据库错误，重试 {retry}，Agent 运行: {agent_run_id}: {str(db_error)}"
                )
                if retry < 2:  # 还不是最后一次重试
                    await asyncio.sleep(0.5 * (2**retry))  # 指数退避
                else:
                    logger.error(
                        f"所有重试后更新 Agent 运行状态失败: {agent_run_id}",
                        exc_info=True,
                    )
                    return False
    except Exception as e:
        logger.error(
            f"更新 Agent 运行状态时发生意外错误 {agent_run_id}: {str(e)}",
            exc_info=True,
        )
        return False

    return False
