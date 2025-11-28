import asyncio
import json
import time
import traceback
import uuid
from datetime import datetime
from typing import Any, Optional

import dramatiq
import structlog
from dramatiq.brokers.redis import RedisBroker
from loguru import logger
from tenacity import (
    AsyncRetrying,
    Retrying,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
)

from configs import app_config
from core.run import run_agent
from core.services import redis
from models.agent_run import AgentRun, AgentRuns

logger.info(f"正在配置 Dramatiq 代理，Redis 地址: {app_config.REDIS_HOST}:{app_config.REDIS_PORT}")
redis_broker = RedisBroker(
    host=app_config.REDIS_HOST,
    port=app_config.REDIS_PORT,
    db=app_config.REDIS_DB,
    password=app_config.REDIS_PASSWORD,
    middleware=[dramatiq.middleware.AsyncIO()],
)

dramatiq.set_broker(redis_broker)

_initialized = False
instance_id = ""


def check_terminating_tool_call(response: dict[str, Any]) -> Optional[str]:
    if response.get("type") != "status":
        return None

    metadata = response.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            metadata = {}

    if not metadata.get("agent_should_terminate"):
        return None

    content = response.get("content", {})
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            content = {}

    if isinstance(content, dict):
        function_name = content.get("function_name")
        if function_name in ["ask", "complete"]:
            return function_name

    return None


async def initialize():
    """使用主 API 的资源初始化 Agent API。"""
    global instance_id, _initialized

    if _initialized:
        return  # 已经初始化

    if not instance_id:
        instance_id = str(uuid.uuid4())[:8]

    logger.info(f"正在初始化工作进程，Redis 地址: {app_config.REDIS_HOST}:{app_config.REDIS_PORT}")
    await redis.initialize_async()

    _initialized = True
    logger.info(f"✅ 工作进程初始化成功，实例 ID: {instance_id}")


@dramatiq.actor
async def check_health(key: str):
    structlog.contextvars.clear_contextvars()
    await redis.set(key, "healthy", ex=redis.REDIS_KEY_TTL)


async def acquire_run_lock(agent_run_id: str, instance_id: str) -> bool:
    run_lock_key = f"agent_run_lock:{agent_run_id}"
    lock_acquired = await redis.set(run_lock_key, instance_id, nx=True, ex=redis.REDIS_KEY_TTL)

    if not lock_acquired:
        existing_instance = await redis.get(run_lock_key)
        existing_instance_str = (
            existing_instance.decode() if isinstance(existing_instance, bytes) else existing_instance or None
        )

        if existing_instance_str:
            instance_active_key = f"active_run:{existing_instance_str}:{agent_run_id}"
            instance_still_alive = await redis.get(instance_active_key)

            db_run_status = None
            try:
                run_result = AgentRuns.get_by_id(agent_run_id, AgentRun.status)
                if run_result:
                    db_run_status = run_result.status
            except Exception as db_err:
                logger.warning("查询 {} 的数据库状态失败：{}", agent_run_id, db_err)

            if instance_still_alive or db_run_status == "running":
                logger.info("agent run {} 正由实例 {} 处理，跳过重复执行。", agent_run_id, existing_instance_str)
                return False
            else:
                logger.warning(
                    "检测到 {} 的锁已失效（持有实例 {} 无响应，数据库状态：{}），尝试获取锁",
                    agent_run_id,
                    existing_instance_str,
                    db_run_status,
                )
                await redis.delete(run_lock_key)
                lock_acquired = await redis.set(run_lock_key, instance_id, nx=True, ex=redis.REDIS_KEY_TTL)
                if not lock_acquired:
                    logger.info("清理旧锁时，其他 worker 已抢先拿到 {} 的锁，跳过。", agent_run_id)
                    return False
        else:
            lock_acquired = await redis.set(run_lock_key, instance_id, nx=True, ex=redis.REDIS_KEY_TTL)
            if not lock_acquired:
                logger.info("agent run {} 已在别的实例执行中，跳过。", agent_run_id)
                return False

    return True


async def send_completion_notification(
    thread_id: str, agent_config: Optional[dict[str, Any]], complete_tool_called: bool
):
    if not complete_tool_called:
        return

    logger.info("已发送任务完成通知")


async def send_failure_notification(thread_id: str, error_message: str):
    logger.info("已发送任务失败通知")


def create_redis_keys(agent_run_id: str, instance_id: str) -> dict[str, str]:
    return {
        "response_list": f"agent_run:{agent_run_id}:responses",
        "response_channel": f"agent_run:{agent_run_id}:new_response",
        "instance_control_channel": f"agent_run:{agent_run_id}:control:{instance_id}",
        "global_control_channel": f"agent_run:{agent_run_id}:control",
        "instance_active": f"active_run:{instance_id}:{agent_run_id}",
    }


async def create_stop_signal_checker(
    pubsub, agent_run_id: str, instance_id: str, instance_active_key: str, cancellation_event: asyncio.Event
):
    stop_signal_received = False
    total_responses = 0

    async def check_for_stop_signal():
        nonlocal stop_signal_received, total_responses
        if not pubsub:
            return
        try:
            while not stop_signal_received:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.5)
                if message and message.get("type") == "message":
                    data = message.get("data")
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")
                    if data == "STOP":
                        logger.debug(f"实例 {instance_id} 收到 agent run {agent_run_id} 的停止指令")
                        stop_signal_received = True
                        cancellation_event.set()
                        break

                if total_responses % 50 == 0:
                    try:
                        await redis.expire(instance_active_key, redis.REDIS_KEY_TTL)
                    except Exception as ttl_err:
                        logger.warning(f"刷新实例存活 TTL 失败：{instance_active_key}，错误：{ttl_err}")
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logger.debug(f"{agent_run_id}（实例 {instance_id}）的停止信号检查器已取消")
        except Exception as e:
            logger.exception(f"{agent_run_id}的停止信号检查器错误")
            stop_signal_received = True

    return check_for_stop_signal, stop_signal_received


async def process_agent_responses(
    agent_gen,
    agent_run_id: str,
    redis_keys: dict[str, str],
    worker_start: float,
    stop_signal_checker_state: dict[str, Any],
) -> tuple[str, Optional[str], bool, int]:
    final_status = "running"
    error_message = None
    first_response_logged = False
    complete_tool_called = False
    total_responses = 0
    pending_redis_operations = []

    async for response in agent_gen:
        if not first_response_logged:
            first_token_time = (time.time() - worker_start) * 1000
            logger.info(f"⏱️ [TIMING] 🎯 从任务开始到收到第一个响应：{first_token_time:.1f}ms")
            first_response_logged = True

        if stop_signal_checker_state.get("stop_signal_received"):
            logger.debug("agent run {} 已被信号终止", agent_run_id)
            final_status = "stopped"
            break

        response_json = json.dumps(response)
        pending_redis_operations.append(asyncio.create_task(redis.rpush(redis_keys["response_list"], response_json)))
        pending_redis_operations.append(asyncio.create_task(redis.publish(redis_keys["response_channel"], "new")))
        total_responses += 1
        stop_signal_checker_state["total_responses"] = total_responses

        terminating_tool = check_terminating_tool_call(response)
        if terminating_tool == "complete":
            complete_tool_called = True
            logger.info(f"agent run {agent_run_id} 已调用 complete 工具")
        elif terminating_tool == "ask":
            logger.debug(f"agent run {agent_run_id} 调用了 ask 工具（流程终止，不通知）")

        if response.get("type") == "status":
            status_val = response.get("status")

            if status_val in ["completed", "failed", "stopped", "error"]:
                logger.info(f"agent run {agent_run_id} 结束，状态：{status_val}")
                final_status = status_val if status_val != "error" else "failed"
                if status_val in ["failed", "stopped", "error"]:
                    error_message = response.get("message", f"Run ended with status: {status_val}")
                    logger.error(f"agent run 失败：{error_message}")
                break

    stop_signal_checker_state["pending_redis_operations"] = pending_redis_operations
    return final_status, error_message, complete_tool_called, total_responses


async def handle_normal_completion(
    agent_run_id: str, start_time: datetime, total_responses: int, redis_keys: dict[str, str]
) -> dict[str, str]:
    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"agent run {agent_run_id} 正常结束（耗时 {duration:.2f} 秒，返回 {total_responses} 条响应）")
    completion_message = {"type": "status", "status": "completed", "message": "agent run 正常结束"}
    await redis.rpush(redis_keys["response_list"], json.dumps(completion_message))
    await redis.publish(redis_keys["response_channel"], "new")
    return completion_message


async def publish_final_control_signal(final_status: str, global_control_channel: str):
    control_signal = "END_STREAM" if final_status == "completed" else "ERROR" if final_status == "failed" else "STOP"
    try:
        await redis.publish(global_control_channel, control_signal)
        logger.debug("已向 {} 发送最终控制信号：'{}'", global_control_channel, control_signal)
    except Exception as e:
        logger.warning("向 {} 发送最终控制信号失败：{}，错误：{}", global_control_channel, control_signal, e)


async def cleanup_pubsub(pubsub, agent_run_id: str):
    if not pubsub:
        return

    pubsub_cleaned = False
    try:
        await pubsub.unsubscribe()
        await pubsub.close()
        pubsub_cleaned = True
        logger.debug(f"{agent_run_id} 的 PubSub 连接已关闭")
    except asyncio.CancelledError:
        if not pubsub_cleaned:
            try:
                await pubsub.unsubscribe()
                await pubsub.close()
                logger.debug("取消场景下，{} 的 PubSub 连接已关闭", agent_run_id)
            except Exception:
                pass
    except Exception as e:
        logger.warning("关闭 {} 的 PubSub 时出错：{}", agent_run_id, e)


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
    worker_start = time.time()
    timings = {}

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        agent_run_id=agent_run_id,
        thread_id=thread_id,
        request_id=request_id,
    )

    logger.info("⏱️ [TIMING] Worker 于 {} 收到任务", worker_start)

    t = time.time()
    try:
        await initialize()
    except Exception as e:
        logger.exception("Redis 连接初始化失败")
        raise e
    timings["initialize"] = (time.time() - t) * 1000

    lock_acquired = await acquire_run_lock(agent_run_id, instance_id)
    if not lock_acquired:
        return

    timings["lock_acquisition"] = (time.time() - worker_start) * 1000 - timings["initialize"]
    logger.info(
        f"⏱️ [TIMING] Worker 初始化: {timings['initialize']:.1f}ms | 获取锁: {timings['lock_acquisition']:.1f}ms"
    )
    logger.info(f"后台启动 agent run：{agent_run_id}，线程：{thread_id}，实例：{instance_id}")

    logger.info("🚀 使用模型: {}", model_name)

    start_time = datetime.now()
    pubsub = None
    stop_checker = None
    pending_redis_operations = []
    cancellation_event = asyncio.Event()

    redis_keys = create_redis_keys(agent_run_id, instance_id)

    try:
        pubsub = await redis.create_pubsub()
        try:
            retry = AsyncRetrying(stop=stop_after_attempt(3), wait=wait_fixed(1))
            async for attempt in retry:
                with attempt:
                    await pubsub.subscribe(
                        redis_keys["instance_control_channel"],
                        redis_keys["global_control_channel"],
                    )
        except Exception as e:
            logger.exception("Redis 订阅控制频道失败")
            raise e

        logger.info(
            "已订阅控制频道：{}，{}",
            redis_keys["instance_control_channel"],
            redis_keys["global_control_channel"],
        )

        stop_signal_checker_state = {
            "stop_signal_received": False,
            "total_responses": 0,
        }
        check_stop_signal_fn, _ = await create_stop_signal_checker(
            pubsub,
            agent_run_id,
            instance_id,
            redis_keys["instance_active"],
            cancellation_event,
        )

        async def check_for_stop_signal_wrapper():
            while not stop_signal_checker_state.get("stop_signal_received"):
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.5)
                if message and message.get("type") == "message":
                    data = message.get("data")
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")
                    if data == "STOP":
                        logger.debug(f"收到 agent run {agent_run_id} 的停止信号（实例：{instance_id}）")
                        stop_signal_checker_state["stop_signal_received"] = True
                        cancellation_event.set()
                        break

                if stop_signal_checker_state.get("total_responses", 0) % 50 == 0:
                    try:
                        await redis.expire(redis_keys["instance_active"], redis.REDIS_KEY_TTL)
                    except Exception as ttl_err:
                        logger.warning("刷新存活 TTL 失败：{}，错误：{}", redis_keys["instance_active"], ttl_err)
                await asyncio.sleep(0.1)

        stop_checker = asyncio.create_task(check_for_stop_signal_wrapper())
        await redis.set(redis_keys["instance_active"], "running", ex=redis.REDIS_KEY_TTL)

        agent_gen = run_agent(
            thread_id=thread_id,
            model_name=model_name,
            project_id=project_id,
            agent_config=agent_config,
            cancellation_event=cancellation_event,
        )

        total_to_ready = (time.time() - worker_start) * 1000
        logger.info(f"⏱️ [TIMING] 从任务开始到第一次 LLM 调用准备就绪：{total_to_ready:.1f}ms")

        (
            final_status,
            error_message,
            complete_tool_called,
            total_responses,
        ) = await process_agent_responses(agent_gen, agent_run_id, redis_keys, worker_start, stop_signal_checker_state)

        pending_redis_operations = stop_signal_checker_state.get("pending_redis_operations", [])

        if final_status == "running":
            final_status = "completed"
            await handle_normal_completion(agent_run_id, start_time, total_responses, redis_keys)
            await send_completion_notification(thread_id, agent_config, complete_tool_called)
            if not complete_tool_called:
                logger.info("agent run {} 未调用 complete 工具即结束，跳过通知。", agent_run_id)

        all_responses_json = await redis.lrange(redis_keys["response_list"], 0, -1)
        all_responses = [json.loads(r) for r in all_responses_json]

        await update_agent_run_status(
            agent_run_id,
            final_status,
            error=error_message,
        )

        if final_status == "failed" and error_message:
            await send_failure_notification(thread_id, error_message)

        await publish_final_control_signal(final_status, redis_keys["global_control_channel"])

    except Exception as e:
        error_message = str(e)
        traceback_str = traceback.format_exc()
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(
            f"agent run {agent_run_id} 运行 {duration:.2f} 秒后报错：{error_message}\n{traceback_str}（实例：{instance_id}）"
        )
        final_status = "failed"

        await send_failure_notification(thread_id, error_message)

        # 将错误消息推送到 Redis 列表
        error_response = {"type": "status", "status": "error", "message": error_message}
        try:
            await redis.rpush(redis_keys["response_list"], json.dumps(error_response))
            await redis.publish(redis_keys["response_channel"], "new")
        except Exception as redis_err:
            logger.error("向 Redis 推送 {} 的错误响应失败：{}", agent_run_id, redis_err)

        # 更新数据库状态
        await update_agent_run_status(agent_run_id, "failed", error=f"{error_message}\n{traceback_str}")

        # 发布 ERROR 信号
        try:
            await redis.publish(redis_keys["global_control_channel"], "ERROR")
            logger.debug("已向 {} 发送 ERROR 信号", redis_keys["global_control_channel"])
        except Exception as e:
            logger.warning("发送 ERROR 信号失败：{}", e)

    finally:
        # 清理停止检查器任务
        if stop_checker and not stop_checker.done():
            stop_checker.cancel()
            try:
                await stop_checker
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning("取消停止检查器时出错：{}", e)

        await cleanup_pubsub(pubsub, agent_run_id)
        await _cleanup_redis_response_list(agent_run_id)
        await _cleanup_redis_instance_key(agent_run_id, instance_id)
        await _cleanup_redis_run_lock(agent_run_id)

        try:
            await asyncio.wait_for(asyncio.gather(*pending_redis_operations), timeout=30.0)
        except TimeoutError:
            logger.warning("等待 {} 的 Redis 操作超时", agent_run_id)

        logger.debug(
            "agent run 后台任务已完全结束：{}（实例：{}），最终状态：{}",
            agent_run_id,
            instance_id,
            final_status,
        )


async def _cleanup_redis_instance_key(agent_run_id: str, instance_id: str):
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
    agent_run_id: str,
    status: str,
    error: Optional[str] = None,
) -> bool:
    """
    更新agent运行状态。
    如果更新成功则返回True。
    """
    try:
        # 使用 tenacity 的 Retrying 进行重试，最多重试 3 次
        for attempt in Retrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, max=10),
            retry_error_callback=lambda retry_state: False,  # 重试失败后返回 False
            reraise=True,  # 重试失败后重新抛出异常
        ):
            with attempt:
                agent_run = AgentRuns.update_status(agent_run_id, status, error)

                if not agent_run:
                    logger.warning(
                        f"数据库更新未返回数据，Agent 运行: {agent_run_id}，重试: {attempt.retry_state.attempt_number}"
                    )
                    raise Exception(f"数据库更新未返回数据: {agent_run_id}")

                # 更新成功，返回 True
                return True

    except Exception as e:
        logger.error("更新 Agent 运行状态失败 {}: {}", agent_run_id, e)
        return False

    return False
