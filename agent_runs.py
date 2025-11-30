import asyncio
import json
import traceback
import uuid
from datetime import datetime
from typing import Any, Optional

import structlog
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from loguru import logger

import core_utils
from core.services import redis
from core.services.db import get_db
from core.utils.auth_utils import (
    get_user_id_from_stream_auth,
    verify_and_get_user_id_from_jwt,
)
from core.utils.project_helpers import generate_and_update_project_name
from core.utils.run_management import stop_agent_run_with_helpers as stop_agent_run
from models.agent_run import AgentRun, AgentRuns
from models.message import Message, Messages
from models.project import Project, Projects
from models.thread import Thread, Threads
from run_agent_background import run_agent_background
from schemas.threads import UnifiedAgentStartResponse

router = APIRouter(tags=["agent-runs"])


async def _get_agent_run_with_access_check(agent_run_id: str, user_id: str):
    """ "
    获取代理运行并验证用户是否有访问权限。
    """
    agent_run = AgentRuns.get_by_id(agent_run_id)
    if not agent_run:
        raise HTTPException(status_code=404, detail="未找到代理运行记录")

    return agent_run.model_dump(mode="json")


async def _get_effective_model(model_name: Optional[str], account_id: str) -> str:
    return "glm-4.6"


async def _create_agent_run_record(
    thread_id: str, effective_model: str, actual_user_id: str, extra_metadata: Optional[dict[str, Any]] = None
) -> str:
    """
    在数据库中创建代理运行

    参数:
        client: 数据库客户端
        thread_id: 关联的线程ID
        agent_config: 代理配置字典
        effective_model: 使用的模型名称
        actual_user_id: 实际发起运行的用户ID
        extra_metadata: 额外的元数据

    返回:
        agent_run_id: 创建的代理运行记录ID
    """
    run_metadata = {"model_name": effective_model, "actual_user_id": actual_user_id}

    if extra_metadata:
        run_metadata.update(extra_metadata)

    current_time = datetime.now()
    agent_run = AgentRun(
        **{
            "thread_id": thread_id,
            "status": "running",
            "started_at": current_time,
            "created_at": current_time,
            "updated_at": current_time,
            "agent_id": None,
            "agent_version_id": None,
            "meta": run_metadata,
        }
    )
    agent_run = AgentRuns.insert(agent_run)

    agent_run_id = agent_run.id
    structlog.contextvars.bind_contextvars(agent_run_id=agent_run_id)
    logger.debug(f"已新建 agent run：{agent_run_id}")

    # 清除运行中缓存
    # try:
    #     from core.runtime_cache import invalidate_running_runs_cache

    #     await invalidate_running_runs_cache(actual_user_id)
    # except Exception as cache_error:
    #     logger.warning(f"清除运行中缓存失败：{cache_error}")

    # 在Redis中注册运行
    instance_key = f"active_run:{core_utils.instance_id}:{agent_run_id}"
    try:
        await redis.set(instance_key, "running", ex=redis.REDIS_KEY_TTL)
    except Exception as e:
        logger.warning("在 Redis 中注册 agent run（{}）失败：{}", instance_key, e)

    return agent_run_id


async def _trigger_agent_background(
    agent_run_id: str,
    thread_id: str,
    project_id: str,
    effective_model: str,
    agent_id: Optional[str] = None,
    account_id: Optional[str] = None,
):
    """
    触发后台代理执行。

    参数:
        agent_run_id: 代理运行记录ID
        thread_id: 线程ID
        project_id: 项目ID
        effective_model: 模型名称
        agent_id: 智能体ID
        account_id: 用户ID
    """
    request_id = structlog.contextvars.get_contextvars().get("request_id")

    try:
        message = run_agent_background.send(
            agent_run_id=agent_run_id,
            thread_id=thread_id,
            instance_id=core_utils.instance_id,
            project_id=project_id,
            model_name=effective_model,
            agent_id=agent_id,
            account_id=account_id,
            request_id=request_id,
        )
        message_id = message.message_id if hasattr(message, "message_id") else "N/A"
        logger.info(f"agent run {agent_run_id} 已成功发送至 Dramatiq 队列（消息 ID：{message_id}）")
    except Exception as e:
        logger.exception(f"agent run {agent_run_id} 发送至 Dramatiq 队列失败")
        # raise HTTPException(status_code=500, detail=f"触发后台代理执行失败：{e}")
        raise


async def start_agent_run(
    account_id: str,
    prompt: str,
    agent_id: Optional[str] = None,
    model_name: Optional[str] = None,
    thread_id: Optional[str] = None,
    project_id: Optional[str] = None,
    message_content: Optional[str] = None,  # 已预处理的内容（含文件引用）
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    启动 agent run 的核心函数

    供以下模块调用：
        HTTP 端点（unified_agent_start）
        触发器执行服务
        任何其他内部调用者

    消息创建规则：
        新线程（thread_id=None）：必须用 prompt 创建一条用户消息
        已有线程 + 提供 prompt：追加一条用户消息
        已有线程 + 无 prompt：不创建消息（假定已通过 /threads/{id}/messages/add 添加）

    支持两种客户端模式：
        单次对话：POST /agent/start 带 prompt → 创建线程 + 消息 + 启动 agent
        多次对话：先 POST /threads/{id}/messages/add，再 POST /agent/start（无 prompt）

    参数:
        account_id：用户账户 ID（必填）
        prompt：用户输入（新线程必填，已有线程可选）
        agent_id：指定 Agent（可选，默认按配置）
        model_name：指定模型（可选，默认按层级的配置）
        thread_id：沿用已有线程（传 None 则新建）
        project_id：已有线程所属项目（若提供 thread_id 则必填）
        message_content：外部已预处理的消息内容（含文件引用）
        metadata：附加的 agent run 元数据
        skip_limits_check：跳过计费/限额校验（预检过的调用者使用）

    返回:
        字典类型，包含 thread_id, agent_run_id, project_id, status
    """
    is_new_thread = thread_id is None

    # 如果传了 message_content 就用它，否则用 prompt
    final_message_content = message_content or prompt

    # 获取有效模型
    effective_model = await _get_effective_model(model_name, account_id)

    if is_new_thread:
        if not project_id:
            project_id = str(uuid.uuid4())
            placeholder_name = f"{prompt[:30]}..." if len(prompt) > 30 else prompt

            current_time = datetime.now()
            project = Projects.insert(
                Project(
                    **{
                        "project_id": project_id,
                        "account_id": account_id,
                        "name": placeholder_name,
                        "created_at": current_time,
                        "updated_at": current_time,
                    }
                )
            )
            project_id = project.project_id

            # 缓存项目信息
            # try:
            #     from core.runtime_cache import set_cached_project_metadata

            #     await set_cached_project_metadata(project_id, {})
            # except Exception:
            #     pass

            # 触发后台命名任务
            asyncio.create_task(generate_and_update_project_name(project_id=project_id, prompt=prompt))

        # 创建新线程
        thread_id = str(uuid.uuid4())
        current_time = datetime.now()
        thread_data = {
            "thread_id": thread_id,
            "project_id": project_id,
            "account_id": account_id,
            "created_at": current_time,
            "updated_at": current_time,
        }
        thread = Threads.insert(Thread(**thread_data))
        thread_id = thread.thread_id
        logger.debug("创建新线程: {}", thread_id)

        structlog.contextvars.bind_contextvars(thread_id=thread_id, project_id=project_id, account_id=account_id)

        # 更新线程计数缓存
        # try:
        #     from core.runtime_cache import increment_thread_count_cache

        #     asyncio.create_task(increment_thread_count_cache(account_id))
        # except Exception:
        #     pass

    # 创建 agent run
    async def create_message():
        """
        消息创建逻辑：
            新线程：必定创建消息（prompt 已在接口层校验为必填）
            已有线程 + 传了 prompt：创建消息（用户想一边加消息一边启动 agent）
            已有线程 + 没传 prompt：跳过（用户已通过 /threads/{id}/messages/add 单独加过消息）

        这样可避免“先加消息再启动”的两步流程出现重复或空消息：
        1. /threads/{id}/messages/add（带上消息）
        2. /agent/start（不再带 prompt，仅启动 agent）
        """
        # 如果没内容，则跳过
        if not final_message_content or not final_message_content.strip():
            return

        # 创建初始用户消息
        message_id = str(uuid.uuid4())
        message_payload = {"role": "user", "content": final_message_content}
        current_time = datetime.now()
        message = Message(
            **{
                "message_id": message_id,
                "thread_id": thread_id,
                "type": "user",
                "is_llm_message": True,
                "content": message_payload,
                "created_at": current_time,
                "updated_at": current_time,
            }
        )
        Messages.save(message)
        logger.debug(f"已为线程 {thread_id} 创建用户消息")

    async def create_agent_run():
        return await _create_agent_run_record(thread_id, effective_model, account_id, metadata)

    _, agent_run_id = await asyncio.gather(create_message(), create_agent_run())

    # 触发后台执行
    await _trigger_agent_background(agent_run_id, thread_id, project_id, effective_model, agent_id, account_id)

    return {"thread_id": thread_id, "agent_run_id": agent_run_id, "project_id": project_id, "status": "running"}


@router.post(
    "/agent/start",
    response_model=UnifiedAgentStartResponse,
    summary="启动代理（统一接口）",
    operation_id="unified_agent_start",
)
async def unified_agent_start(
    thread_id: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    model_name: Optional[str] = Form(None),
    agent_id: Optional[str] = Form(None),
    files: list[UploadFile] = File(default=[]),
    user_id: str = Depends(verify_and_get_user_id_from_jwt),
):
    """
    统一接口启动代理运行。

    - 如果提供 thread_id：在现有线程上启动代理（可选用户输入和文件）
    - 如果未提供 thread_id：创建新项目和线程并启动代理

    支持新线程和现有线程的文件上传。
    """
    if not core_utils.instance_id:
        raise HTTPException(status_code=500, detail="代理API服务端未初始化实例ID")

    account_id = user_id

    # 调试日志 - 记录接收到的参数
    logger.debug(
        f"接收到代理启动请求: thread_id={thread_id!r}, prompt={prompt[:100] if prompt else None!r}, model_name={model_name!r}, agent_id={agent_id!r}, files_count={len(files)}"
    )
    logger.debug(
        f"参数类型: thread_id={type(thread_id)}, prompt={type(prompt)}, model_name={type(model_name)}, agent_id={type(agent_id)}"
    )

    # 额外的验证日志
    if not thread_id and (not prompt or not prompt.strip()):
        error_msg = f"验证错误: 新线程需要提供用户输入。接收到: prompt={prompt!r} (类型={type(prompt)}), thread_id={thread_id!r}"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail="创建新线程时需要提供用户输入")

    try:
        project_id = None
        message_content = prompt or ""

        if thread_id:
            structlog.contextvars.bind_contextvars(thread_id=thread_id)

            # 验证线程存在并获取元数据
            thread_data = Threads.get_by_id(thread_id, Thread.project_id, Thread.account_id, Thread.meta)

            if not thread_data:
                raise HTTPException(status_code=404, detail="未找到线程")

            project_id = thread_data.project_id or ""
            thread_account_id = thread_data.account_id or ""
            thread_metadata = thread_data.meta

            structlog.contextvars.bind_contextvars(
                project_id=project_id,
                account_id=thread_account_id,
                thread_metadata=thread_metadata,
            )

            # # 获取有效模型
            # effective_model = await _get_effective_model(model_name, account_id)

            # if prompt:
            #     # 没有文件但提供了用户输入 - 创建用户消息
            #     message_id = str(uuid.uuid4())
            #     message_payload = {"role": "user", "content": prompt}
            #     current_time = datetime.now()
            #     message = Message(
            #         **{
            #             "message_id": message_id,
            #             "thread_id": thread_id,
            #             "type": "user",
            #             "is_llm_message": True,
            #             "content": message_payload,
            #             "created_at": current_time,
            #             "updated_at": current_time,
            #         }
            #     )
            #     Messages.save(message)
            #     logger.debug(f"为线程 {thread_id} 创建用户消息")

            # # 创建代理运行记录
            # agent_run_id = await _create_agent_run_record(thread_id, effective_model, account_id)

            # # 触发后台执行
            # await _trigger_agent_background(agent_run_id, thread_id, project_id, effective_model)

            # return {
            #     "thread_id": thread_id,
            #     "agent_run_id": agent_run_id,
            #     "status": "running",
            # }

        else:
            # # 验证新线程是否提供了用户输入
            # if not prompt or (isinstance(prompt, str) and not prompt.strip()):
            #     logger.error(f"验证失败: 新线程需要提供用户输入。接收到 prompt={prompt!r}, 类型={type(prompt)}")
            #     raise HTTPException(
            #         status_code=400,
            #         detail="创建新线程时需要提供用户输入",
            #     )

            # logger.debug(f"使用用户输入和 {len(files)} 个文件创建新线程")

            # # 获取有效模型
            # effective_model = await _get_effective_model(model_name, account_id)

            # # 创建项目
            # placeholder_name = f"{prompt[:30]}..." if len(prompt) > 30 else prompt

            # current_time = datetime.now()
            # project = Projects.insert(
            #     Project(
            #         **{
            #             "project_id": str(uuid.uuid4()),
            #             "account_id": account_id,
            #             "name": placeholder_name,
            #             "created_at": current_time,
            #             "updated_at": current_time,
            #         }
            #     )
            # )

            # project_id = project.project_id
            # logger.info("创建新项目: {}", project_id)

            # # 创建线程
            # current_time = datetime.now()
            # thread_data = {
            #     "thread_id": str(uuid.uuid4()),
            #     "project_id": project_id,
            #     "account_id": account_id,
            #     "created_at": current_time,
            #     "updated_at": current_time,
            # }

            # structlog.contextvars.bind_contextvars(
            #     thread_id=thread_data["thread_id"],
            #     project_id=project_id,
            #     account_id=account_id,
            # )

            # thread = Threads.insert(Thread(**thread_data))
            # thread_id = thread.thread_id
            # logger.debug("创建新线程: {}", thread_id)

            # # 触发后台命名任务
            # asyncio.create_task(generate_and_update_project_name(project_id=project_id, prompt=prompt))

            # # 处理文件上传并创建用户消息
            # message_content = prompt

            # # 创建初始用户消息
            # message_id = str(uuid.uuid4())
            # message_payload = {"role": "user", "content": message_content}
            # current_time = datetime.now()
            # message = Message(
            #     **{
            #         "message_id": message_id,
            #         "thread_id": thread_id,
            #         "type": "user",
            #         "is_llm_message": True,
            #         "content": message_payload,
            #         "created_at": current_time,
            #         "updated_at": current_time,
            #     }
            # )
            # Messages.save(message)

            # # 创建代理运行记录
            # agent_run_id = await _create_agent_run_record(thread_id, effective_model, account_id)

            # # 触发后台执行
            # await _trigger_agent_background(agent_run_id, thread_id, project_id, effective_model, agent_id, account_id)
            pass

        result = await start_agent_run(
            account_id=account_id,
            prompt=prompt or "",
            agent_id=agent_id,
            model_name=model_name,
            thread_id=thread_id,
            project_id=project_id,
            message_content=message_content,
        )

        return {"thread_id": result["thread_id"], "agent_run_id": result["agent_run_id"], "status": "running"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("统一 agent 启动失败")
        # 记录实际错误详情用于调试
        error_details = {
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
        }
        logger.error(f"完整错误详情: {error_details}")
        raise HTTPException(status_code=500, detail=f"智能体启动失败: {str(e)}")


@router.post(
    "/agent-run/{agent_run_id}/stop",
    summary="停止运行中的代理",
    operation_id="stop_agent_run",
)
async def stop_agent(agent_run_id: str, user_id: str = Depends(verify_and_get_user_id_from_jwt)):
    """停止正在运行的代理。"""
    structlog.contextvars.bind_contextvars(
        agent_run_id=agent_run_id,
    )
    logger.debug(f"接收到停止代理运行的请求: {agent_run_id}")
    await _get_agent_run_with_access_check(agent_run_id, user_id)
    await stop_agent_run(agent_run_id)
    return {"status": "stopped"}


@router.get(
    "/agent-runs/active",
    summary="列出所有正在运行的代理",
    operation_id="list_active_agent_runs",
)
async def get_active_agent_runs(
    user_id: str = Depends(verify_and_get_user_id_from_jwt),
):
    """获取当前用户所有线程中的正在运行的代理。"""
    try:
        logger.debug(f"获取用户所有正在运行的代理: {user_id}")

        # 查询所有运行中的代理运行，其中线程属于该用户
        user_threads = AgentRuns.get_running_agent_runs(
            AgentRun.id, AgentRun.thread_id, AgentRun.status, AgentRun.started_at
        )

        if not user_threads:
            return {"active_runs": []}

        # 过滤代理运行，仅包含用户有访问权限的
        # 获取thread_ids并检查访问权限
        thread_ids = [str(run.thread_id) for run in user_threads if run.thread_id and run.thread_id.strip()]
        if not thread_ids:
            logger.debug(f"未找到用户 {user_id} 的任何有效 thread_id")
            return {"active_runs": []}

        # 获取属于用户的线程
        threads = Threads.get_by_ids(thread_ids, user_id)

        if not threads:
            return {"active_runs": []}

        # 创建可访问线程ID的集合
        accessible_thread_ids = {str(thread.thread_id) for thread in threads}

        # 过滤代理运行，仅包含可访问的
        accessible_runs = [
            {
                "id": str(run.id),
                "thread_id": str(run.thread_id),
                "status": run.status,
                "started_at": run.started_at,
            }
            for run in user_threads
            if run.thread_id in accessible_thread_ids
        ]

        logger.debug(f"为用户 {user_id} 找到 {len(accessible_runs)} 个活跃 agent run")
        return {"active_runs": accessible_runs}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取用户 {user_id} 的活跃 agent run 时出错：{str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"获取活跃代理运行记录失败: {str(e)}")


@router.get(
    "/thread/{thread_id}/agent-runs",
    summary="列出线程的代理运行记录",
    operation_id="list_thread_agent_runs",
)
async def get_agent_runs(thread_id: str, user_id: str = Depends(verify_and_get_user_id_from_jwt)):
    """获取线程的所有代理运行记录。"""
    structlog.contextvars.bind_contextvars(
        thread_id=thread_id,
    )
    logger.debug(f"获取线程的代理运行记录: {thread_id}")
    # await verify_and_authorize_thread_access(thread_id, user_id)
    with get_db() as db:
        agent_runs = (
            db.query(
                AgentRun.id,
                AgentRun.thread_id,
                AgentRun.status,
                AgentRun.started_at,
                AgentRun.completed_at,
                AgentRun.error,
                AgentRun.created_at,
                AgentRun.updated_at,
            )
            .filter(AgentRun.thread_id == thread_id)
            .order_by(AgentRun.created_at.desc())
            .all()
        )
    logger.debug(f"找到 {len(agent_runs)} 个线程的代理运行记录: {thread_id}")
    return {"agent_runs": agent_runs}


@router.get(
    "/agent-run/{agent_run_id}",
    summary="获取代理运行记录",
    operation_id="get_agent_run",
)
async def get_agent_run(agent_run_id: str, user_id: str = Depends(verify_and_get_user_id_from_jwt)):
    """获取代理运行状态和响应。"""
    structlog.contextvars.bind_contextvars(
        agent_run_id=agent_run_id,
    )
    logger.debug(f"获取代理运行详情: {agent_run_id}")
    agent_run_data = await _get_agent_run_with_access_check(agent_run_id, user_id)
    # 注意: 响应默认不包含在此，它们在流或数据库中
    return {
        "id": agent_run_data["id"],
        "threadId": agent_run_data["thread_id"],
        "status": agent_run_data["status"],
        "startedAt": agent_run_data["started_at"],
        "completedAt": agent_run_data["completed_at"],
        "error": agent_run_data["error"],
    }


@router.get(
    "/agent-run/{agent_run_id}/stream",
    summary="流式代理运行",
    operation_id="stream_agent_run",
)
async def stream_agent_run(agent_run_id: str, token: Optional[str] = None, request: Request = None):
    """使用Redis列表和发布/订阅流式传输代理运行的响应。"""
    logger.debug(f"开始代理运行的流式传输: {agent_run_id}")

    user_id = await get_user_id_from_stream_auth(request, token)  # 实际上瞬间完成
    agent_run_data = await _get_agent_run_with_access_check(agent_run_id, user_id)

    structlog.contextvars.bind_contextvars(
        agent_run_id=agent_run_id,
        user_id=user_id,
    )

    response_list_key = f"agent_run:{agent_run_id}:responses"
    response_channel = f"agent_run:{agent_run_id}:new_response"
    control_channel = f"agent_run:{agent_run_id}:control"  # 全局控制通道

    async def stream_generator(agent_run_data):
        logger.debug(f"通过 Redis 列表 {response_list_key} 与频道 {response_channel} 为 {agent_run_id} 推送流式响应")
        last_processed_index = -1
        # 单个pubsub用于响应+控制
        listener_task = None
        terminate_stream = False
        initial_yield_complete = False

        try:
            # 1. 从Redis列表获取并发送初始响应
            initial_responses_json = await redis.lrange(response_list_key, 0, -1)
            initial_responses = []
            if initial_responses_json:
                initial_responses = [json.loads(r) for r in initial_responses_json]
                logger.debug(f"为 {agent_run_id} 发送 {len(initial_responses)} 个初始响应")
                for response in initial_responses:
                    yield f"data: {json.dumps(response)}\n\n"
                last_processed_index = len(initial_responses) - 1
            initial_yield_complete = True

            # 2. 检查运行状态
            current_status = agent_run_data.get("status") if agent_run_data else None

            if current_status != "running":
                logger.debug(f"代理运行 {agent_run_id} 未在运行（状态: {current_status}），结束流式传输。")
                yield f"data: {json.dumps({'type': 'status', 'status': 'completed'})}\n\n"
                return

            structlog.contextvars.bind_contextvars(
                thread_id=agent_run_data.get("thread_id"),
            )

            # 3. 使用单个Pub/Sub连接订阅两个通道
            pubsub = await redis.create_pubsub()
            await pubsub.subscribe(response_channel, control_channel)
            logger.debug(f"已订阅通道: {response_channel}, {control_channel}")

            # 用于监听器和主生成器循环之间通信的队列
            message_queue = asyncio.Queue()

            async def listen_messages():
                listener = pubsub.listen()
                task = asyncio.create_task(listener.__anext__())

                while not terminate_stream:
                    done, _ = await asyncio.wait([task], return_when=asyncio.FIRST_COMPLETED)
                    for finished in done:
                        try:
                            message = finished.result()
                            if message and isinstance(message, dict) and message.get("type") == "message":
                                channel = message.get("channel")
                                data = message.get("data")
                                if isinstance(data, bytes):
                                    data = data.decode("utf-8")

                                if channel == response_channel and data == "new":
                                    await message_queue.put({"type": "new_response"})
                                elif channel == control_channel and data in [
                                    "STOP",
                                    "END_STREAM",
                                    "ERROR",
                                ]:
                                    logger.debug(f"接收到 {agent_run_id} 的控制信号 '{data}'")
                                    await message_queue.put({"type": "control", "data": data})
                                    return  # 收到控制信号时停止监听

                        except StopAsyncIteration:
                            logger.warning(f"{agent_run_id} 的监听器已停止。")
                            await message_queue.put(
                                {
                                    "type": "error",
                                    "data": "监听器意外停止",
                                }
                            )
                            return
                        except Exception as e:
                            logger.error(f"{agent_run_id} 的监听器出错：{e}")
                            await message_queue.put({"type": "error", "data": "监听器故障"})
                            return
                        finally:
                            # 如果继续则重新订阅下一条消息
                            if not terminate_stream:
                                task = asyncio.create_task(listener.__anext__())

            listener_task = asyncio.create_task(listen_messages())

            # 4. 主循环处理队列中的消息
            while not terminate_stream:
                try:
                    queue_item = await message_queue.get()

                    if queue_item["type"] == "new_response":
                        # 从 Redis 列表中抓取自上次处理位置之后的新响应
                        new_start_index = last_processed_index + 1
                        new_responses_json = await redis.lrange(response_list_key, new_start_index, -1)

                        if new_responses_json:
                            new_responses = [json.loads(r) for r in new_responses_json]
                            num_new = len(new_responses)
                            # logger.debug(f"收到 {agent_run_id} 的 {num_new} 条新响应（自索引 {new_start_index} 起）")
                            for response in new_responses:
                                yield f"data: {json.dumps(response)}\n\n"
                                # 检查该响应是否标识任务结束
                                if response.get("type") == "status" and response.get("status") in [
                                    "completed",
                                    "failed",
                                    "stopped",
                                ]:
                                    logger.debug(f"流式数据中检测到状态变更，任务结束：{response.get('status')}")
                                    terminate_stream = True
                                    break  # 停止继续处理后续新响应
                            last_processed_index += num_new
                        if terminate_stream:
                            break

                    elif queue_item["type"] == "control":
                        control_signal = queue_item["data"]
                        terminate_stream = True  # 一旦收到任何控制信号，立即停止流式推送
                        yield f"data: {json.dumps({'type': 'status', 'status': control_signal})}\n\n"
                        break

                    elif queue_item["type"] == "error":
                        logger.error(f"{agent_run_id} 监听器报错：{queue_item['data']}")
                        terminate_stream = True
                        yield f"data: {json.dumps({'type': 'status', 'status': 'error'})}\n\n"
                        break

                except asyncio.CancelledError:
                    logger.debug(f"{agent_run_id} 的流式生成主循环已取消")
                    terminate_stream = True
                    break
                except Exception as loop_err:
                    logger.exception(f"{agent_run_id} 的流式生成主循环出错")
                    terminate_stream = True
                    yield f"data: {json.dumps({'type': 'status', 'status': 'error', 'message': f'流式推送失败: {loop_err}'})}\n\n"
                    break

        except Exception as e:
            logger.exception(f"为 agent run {agent_run_id} 初始化流式推送失败")
            # 仅在初始发送未发生时发送错误
            if not initial_yield_complete:
                yield f"data: {json.dumps({'type': 'status', 'status': 'error', 'message': f'启动流失败: {e}'})}\n\n"
        finally:
            terminate_stream = True
            # 优雅关闭顺序: 取消订阅 → 关闭 → 取消
            # 即使任务被取消，也要确保清理工作完成。
            pubsub_cleaned = False
            try:
                if "pubsub" in locals() and pubsub:
                    await pubsub.unsubscribe(response_channel, control_channel)
                    await pubsub.close()
                    pubsub_cleaned = True
                    logger.debug("已清理 {} 的 PubSub", agent_run_id)
            except asyncio.CancelledError:
                # 即使在取消时也要尝试清理
                if "pubsub" in locals() and pubsub and not pubsub_cleaned:
                    try:
                        await pubsub.unsubscribe(response_channel, control_channel)
                        await pubsub.close()
                        logger.debug(f"{agent_run_id} 已取消，相关 PubSub 已清理完毕")
                    except Exception:
                        pass  # 忽略取消清理时的错误
            except Exception as e:
                logger.warning("清理 {} 的 PubSub 时出错：{}", agent_run_id, e)

            if listener_task:
                listener_task.cancel()
                try:
                    await listener_task  # 回收内部任务并忽略其错误
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.debug("listener_task 因 {} 退出", e)
            # 短暂等待任务取消
            await asyncio.sleep(0.1)
            logger.debug("agent run {} 的流式清理已完成", agent_run_id)

    return StreamingResponse(
        stream_generator(agent_run_data),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream",
            "Access-Control-Allow-Origin": "*",
        },
    )
