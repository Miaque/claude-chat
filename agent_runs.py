import asyncio
import json
import os
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

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
from pydantic import BaseModel

import core_utils
from core.services import redis
from core.services.db import get_db
from core.utils.auth_utils import (
    get_user_id_from_stream_auth,
    verify_and_get_user_id_from_jwt,
)
from core.utils.project_helpers import generate_and_update_project_name
from core.utils.run_management import stop_agent_run_with_helpers as stop_agent_run
from models.agent_run import AgentRun, AgentRunModel
from models.message import Message
from models.project import Project
from models.thread import Thread, ThreadModel
from run_agent_background import run_agent_background

router = APIRouter(tags=["agent-runs"])


class UnifiedAgentStartResponse(BaseModel):
    """Unified response model for agent start (both new and existing threads)."""

    thread_id: str
    agent_run_id: str
    status: str = "running"


class AgentVersionResponse(BaseModel):
    """Response model for agent version information."""

    version_id: str
    agent_id: str
    version_number: int
    version_name: str
    system_prompt: str
    model: Optional[str] = None
    configured_mcps: List[Dict[str, Any]]
    custom_mcps: List[Dict[str, Any]]
    agentpress_tools: Dict[str, Any]
    is_active: bool
    created_at: str
    updated_at: str
    created_by: Optional[str] = None


class AgentResponse(BaseModel):
    """Response model for agent information."""

    agent_id: str
    name: str
    description: Optional[str] = None
    system_prompt: Optional[str] = (
        None  # Optional for list operations where config not loaded
    )
    model: Optional[str] = None
    configured_mcps: List[Dict[str, Any]]
    custom_mcps: List[Dict[str, Any]]
    agentpress_tools: Dict[str, Any]
    is_default: bool
    is_public: Optional[bool] = False
    tags: Optional[List[str]] = []
    icon_name: Optional[str] = None
    icon_color: Optional[str] = None
    icon_background: Optional[str] = None
    created_at: str
    updated_at: Optional[str] = None
    current_version_id: Optional[str] = None
    version_count: Optional[int] = 1
    current_version: Optional[AgentVersionResponse] = None
    metadata: Optional[Dict[str, Any]] = None
    account_id: Optional[str] = (
        None  # Internal field, may not always be needed in response
    )


class ThreadAgentResponse(BaseModel):
    """Response model for thread agent information."""

    agent: Optional[AgentResponse]
    source: str
    message: str


async def _get_agent_run_with_access_check(agent_run_id: str, user_id: str):
    """
    Get an agent run and verify the user has access to it.

    Internal helper for this module only.
    """

    with get_db() as db:
        agent_run = db.query(AgentRun).filter(AgentRun.id == agent_run_id).first()

    if not agent_run:
        raise HTTPException(status_code=404, detail="Agent run not found")

    agent_run_data = AgentRunModel.model_validate(agent_run)
    return agent_run_data.model_dump()


async def _get_effective_model(model_name: Optional[str], account_id: str) -> str:
    return "glm-4.6"


async def _create_agent_run_record(thread_id: str, effective_model: str) -> str:
    """
    Create an agent run record in the database.

    Args:
        client: Database client
        thread_id: Thread ID to associate with
        agent_config: Agent configuration dict
        effective_model: Model name to use

    Returns:
        agent_run_id: The created agent run ID
    """
    with get_db() as db:
        agent_run = AgentRun(
            **{
                "thread_id": thread_id,
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "agent_id": None,
                "agent_version_id": None,
                "meta": {"model_name": effective_model},
            }
        )
        db.add(agent_run)
        db.commit()
        db.refresh(agent_run)

    agent_run_id = str(agent_run.id)
    structlog.contextvars.bind_contextvars(agent_run_id=agent_run_id)
    logger.debug(f"Created new agent run: {agent_run_id}")

    # Register run in Redis
    instance_key = f"active_run:{core_utils.instance_id}:{agent_run_id}"
    try:
        await redis.set(instance_key, "running", ex=redis.REDIS_KEY_TTL)
    except Exception as e:
        logger.warning(
            f"Failed to register agent run in Redis ({instance_key}): {str(e)}"
        )

    return agent_run_id


async def _trigger_agent_background(
    agent_run_id: str,
    thread_id: str,
    project_id: str,
    effective_model: str,
    agent_config: Optional[dict],
):
    """
    Trigger the background agent execution.

    Args:
        agent_run_id: Agent run ID
        thread_id: Thread ID
        project_id: Project ID
        effective_model: Model name to use
        agent_config: Agent configuration dict
    """
    request_id = structlog.contextvars.get_contextvars().get("request_id")

    run_agent_background.send(
        agent_run_id=agent_run_id,
        thread_id=thread_id,
        instance_id=core_utils.instance_id,
        project_id=project_id,
        model_name=effective_model,
        agent_config=agent_config,
        request_id=request_id,
    )


@router.post(
    "/agent/start",
    response_model=UnifiedAgentStartResponse,
    summary="Start Agent (Unified)",
    operation_id="unified_agent_start",
)
async def unified_agent_start(
    thread_id: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    model_name: Optional[str] = Form(None),
    agent_id: Optional[str] = Form(None),
    files: List[UploadFile] = File(default=[]),
    user_id: str = Depends(verify_and_get_user_id_from_jwt),
):
    """
    Unified endpoint to start an agent run.

    - If thread_id is provided: Starts agent on existing thread (with optional prompt and files)
    - If thread_id is NOT provided: Creates new project/thread and starts agent

    Supports file uploads for both new and existing threads.
    """
    if not core_utils.instance_id:
        raise HTTPException(
            status_code=500, detail="Agent API not initialized with instance ID"
        )

    account_id = user_id

    # Debug logging - log what we received
    logger.debug(
        f"Received agent start request: thread_id={thread_id!r}, prompt={prompt[:100] if prompt else None!r}, model_name={model_name!r}, agent_id={agent_id!r}, files_count={len(files)}"
    )
    logger.debug(
        f"Parameter types: thread_id={type(thread_id)}, prompt={type(prompt)}, model_name={type(model_name)}, agent_id={type(agent_id)}"
    )

    # Additional validation logging
    if not thread_id and (
        not prompt or (isinstance(prompt, str) and not prompt.strip())
    ):
        error_msg = f"VALIDATION ERROR: New thread requires prompt. Received: prompt={prompt!r} (type={type(prompt)}), thread_id={thread_id!r}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=400, detail="prompt is required when creating a new thread"
        )

    try:
        if thread_id:
            logger.debug(f"Starting agent on existing thread: {thread_id}")
            structlog.contextvars.bind_contextvars(thread_id=thread_id)

            # Validate thread exists and get metadata
            with get_db() as db:
                thread_result = (
                    db.query(Thread.project_id, Thread.account_id, Thread.meta)
                    .filter(Thread.thread_id == thread_id)
                    .first()
                )

            if not thread_result:
                raise HTTPException(status_code=404, detail="Thread not found")

            thread_data = ThreadModel.model_validate(thread_result)
            project_id = str(thread_data.project_id)
            thread_account_id = str(thread_data.account_id)
            thread_metadata = thread_data.meta

            structlog.contextvars.bind_contextvars(
                project_id=project_id,
                account_id=thread_account_id,
                thread_metadata=thread_metadata,
            )

            # Get effective model
            effective_model = await _get_effective_model(model_name, thread_account_id)

            if prompt:
                # No files, but prompt provided - create user message
                message_id = str(uuid.uuid4())
                message_payload = {"role": "user", "content": prompt}
                with get_db() as db:
                    message = Message(
                        **{
                            "message_id": message_id,
                            "thread_id": thread_id,
                            "type": "user",
                            "is_llm_message": True,
                            "content": message_payload,
                            "created_at": datetime.now(),
                        }
                    )
                    db.add(message)
                    db.commit()

                logger.debug(f"Created user message for thread {thread_id}")

            # Create agent run
            agent_run_id = await _create_agent_run_record(thread_id, effective_model)

            # Trigger background execution
            await _trigger_agent_background(
                agent_run_id, thread_id, project_id, effective_model, agent_config={}
            )

            return {
                "thread_id": thread_id,
                "agent_run_id": agent_run_id,
                "status": "running",
            }

        else:
            # Validate that prompt is provided for new threads
            if not prompt or (isinstance(prompt, str) and not prompt.strip()):
                logger.error(
                    f"Validation failed: prompt is required for new threads. Received prompt={prompt!r}, type={type(prompt)}"
                )
                raise HTTPException(
                    status_code=400,
                    detail="prompt is required when creating a new thread",
                )

            logger.debug(f"Creating new thread with prompt and {len(files)} files")

            # Get effective model
            effective_model = await _get_effective_model(model_name, account_id)

            # Create Project
            placeholder_name = f"{prompt[:30]}..." if len(prompt) > 30 else prompt
            with get_db() as db:
                current_time = datetime.now()
                project = Project(
                    **{
                        "project_id": str(uuid.uuid4()),
                        "account_id": account_id,
                        "name": placeholder_name,
                        "created_at": current_time,
                        "updated_at": current_time,
                    }
                )
                db.add(project)
                db.commit()
                db.refresh(project)

            project_id = str(project.project_id)
            logger.info(f"Created new project: {project_id}")

            # Create Thread
            current_time = datetime.now()
            thread_data = {
                "thread_id": str(uuid.uuid4()),
                "project_id": project_id,
                "account_id": account_id,
                "created_at": current_time,
                "updated_at": current_time,
            }

            structlog.contextvars.bind_contextvars(
                thread_id=thread_data["thread_id"],
                project_id=project_id,
                account_id=account_id,
            )

            with get_db() as db:
                thread = Thread(**thread_data)
                db.add(thread)
                db.commit()
                db.refresh(thread)

            thread_id = str(thread.thread_id)
            logger.debug(f"Created new thread: {thread_id}")

            # Trigger background naming task
            asyncio.create_task(
                generate_and_update_project_name(project_id=project_id, prompt=prompt)
            )

            # Handle file uploads and create user message
            message_content = prompt

            # Create initial user message
            message_id = str(uuid.uuid4())
            message_payload = {"role": "user", "content": message_content}
            with get_db() as db:
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
                db.add(message)
                db.commit()

            # Create agent run
            agent_run_id = await _create_agent_run_record(thread_id, effective_model)

            # Trigger background execution
            await _trigger_agent_background(
                agent_run_id, thread_id, project_id, effective_model, agent_config={}
            )

            return {
                "thread_id": thread_id,
                "agent_run_id": agent_run_id,
                "status": "running",
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error in unified agent start: {str(e)}\n{traceback.format_exc()}"
        )
        # Log the actual error details for debugging
        error_details = {
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
        }
        logger.error(f"Full error details: {error_details}")
        raise HTTPException(status_code=500, detail=f"Failed to start agent: {str(e)}")


@router.post(
    "/agent-run/{agent_run_id}/stop",
    summary="Stop Agent Run",
    operation_id="stop_agent_run",
)
async def stop_agent(
    agent_run_id: str, user_id: str = Depends(verify_and_get_user_id_from_jwt)
):
    """Stop a running agent."""
    structlog.contextvars.bind_contextvars(
        agent_run_id=agent_run_id,
    )
    logger.debug(f"Received request to stop agent run: {agent_run_id}")
    await _get_agent_run_with_access_check(agent_run_id, user_id)
    await stop_agent_run(agent_run_id)
    return {"status": "stopped"}


@router.get(
    "/agent-runs/active",
    summary="List All Active Agent Runs",
    operation_id="list_active_agent_runs",
)
async def get_active_agent_runs(
    user_id: str = Depends(verify_and_get_user_id_from_jwt),
):
    """Get all active (running) agent runs for the current user across all threads."""
    try:
        logger.debug(f"Fetching all active agent runs for user: {user_id}")

        # Query all running agent runs where the thread belongs to the user
        # Join with threads table to filter by account_id
        with get_db() as db:
            agent_runs = (
                db.query(
                    AgentRun.id,
                    AgentRun.thread_id,
                    AgentRun.status,
                    AgentRun.started_at,
                )
                .filter(AgentRun.status == "running")
                .all()
            )

        if not agent_runs:
            return {"active_runs": []}

        # Filter agent runs to only include those from threads the user has access to
        # Get thread_ids and check access
        thread_ids = [run.thread_id for run in agent_runs]

        # Get threads that belong to the user
        with get_db() as db:
            threads = (
                db.query(Thread)
                .filter(Thread.thread_id.in_(thread_ids))
                .filter(Thread.account_id == user_id)
                .all()
            )

        # Create a set of accessible thread IDs
        accessible_thread_ids = {thread.thread_id for thread in threads}

        # Filter agent runs to only include accessible ones
        accessible_runs = [
            {
                "id": run.id,
                "thread_id": run.thread_id,
                "status": run.status,
                "started_at": run.started_at,
            }
            for run in agent_runs
            if run.thread_id in accessible_thread_ids
        ]

        logger.debug(
            f"Found {len(accessible_runs)} active agent runs for user: {user_id}"
        )
        return {"active_runs": accessible_runs}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error fetching active agent runs for user {user_id}: {str(e)}\n{traceback.format_exc()}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch active agent runs: {str(e)}"
        )


@router.get(
    "/thread/{thread_id}/agent-runs",
    summary="List Thread Agent Runs",
    operation_id="list_thread_agent_runs",
)
async def get_agent_runs(
    thread_id: str, user_id: str = Depends(verify_and_get_user_id_from_jwt)
):
    """Get all agent runs for a thread."""
    structlog.contextvars.bind_contextvars(
        thread_id=thread_id,
    )
    logger.debug(f"Fetching agent runs for thread: {thread_id}")
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
    logger.debug(f"Found {len(agent_runs)} agent runs for thread: {thread_id}")
    return {"agent_runs": agent_runs}


@router.get(
    "/agent-run/{agent_run_id}", summary="Get Agent Run", operation_id="get_agent_run"
)
async def get_agent_run(
    agent_run_id: str, user_id: str = Depends(verify_and_get_user_id_from_jwt)
):
    """Get agent run status and responses."""
    structlog.contextvars.bind_contextvars(
        agent_run_id=agent_run_id,
    )
    logger.debug(f"Fetching agent run details: {agent_run_id}")
    agent_run_data = await _get_agent_run_with_access_check(agent_run_id, user_id)
    # Note: Responses are not included here by default, they are in the stream or DB
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
    summary="Stream Agent Run",
    operation_id="stream_agent_run",
)
async def stream_agent_run(
    agent_run_id: str, token: Optional[str] = None, request: Request = None
):
    """Stream the responses of an agent run using Redis Lists and Pub/Sub."""
    logger.debug(f"Starting stream for agent run: {agent_run_id}")

    user_id = await get_user_id_from_stream_auth(request, token)  # practically instant
    agent_run_data = await _get_agent_run_with_access_check(
        agent_run_id, user_id
    )  # 1 db query

    structlog.contextvars.bind_contextvars(
        agent_run_id=agent_run_id,
        user_id=user_id,
    )

    response_list_key = f"agent_run:{agent_run_id}:responses"
    response_channel = f"agent_run:{agent_run_id}:new_response"
    control_channel = f"agent_run:{agent_run_id}:control"  # Global control channel

    async def stream_generator(agent_run_data):
        logger.debug(
            f"Streaming responses for {agent_run_id} using Redis list {response_list_key} and channel {response_channel}"
        )
        last_processed_index = -1
        # Single pubsub used for response + control
        listener_task = None
        terminate_stream = False
        initial_yield_complete = False

        try:
            # 1. Fetch and yield initial responses from Redis list
            initial_responses_json = await redis.lrange(response_list_key, 0, -1)
            initial_responses = []
            if initial_responses_json:
                initial_responses = [json.loads(r) for r in initial_responses_json]
                logger.debug(
                    f"Sending {len(initial_responses)} initial responses for {agent_run_id}"
                )
                for response in initial_responses:
                    yield f"data: {json.dumps(response)}\n\n"
                last_processed_index = len(initial_responses) - 1
            initial_yield_complete = True

            # 2. Check run status
            current_status = agent_run_data.get("status") if agent_run_data else None

            if current_status != "running":
                logger.debug(
                    f"Agent run {agent_run_id} is not running (status: {current_status}). Ending stream."
                )
                yield f"data: {json.dumps({'type': 'status', 'status': 'completed'})}\n\n"
                return

            structlog.contextvars.bind_contextvars(
                thread_id=agent_run_data.get("thread_id"),
            )

            # 3. Use a single Pub/Sub connection subscribed to both channels
            pubsub = await redis.create_pubsub()
            await pubsub.subscribe(response_channel, control_channel)
            logger.debug(
                f"Subscribed to channels: {response_channel}, {control_channel}"
            )

            # Queue to communicate between listeners and the main generator loop
            message_queue = asyncio.Queue()

            async def listen_messages():
                listener = pubsub.listen()
                task = asyncio.create_task(listener.__anext__())

                while not terminate_stream:
                    done, _ = await asyncio.wait(
                        [task], return_when=asyncio.FIRST_COMPLETED
                    )
                    for finished in done:
                        try:
                            message = finished.result()
                            if (
                                message
                                and isinstance(message, dict)
                                and message.get("type") == "message"
                            ):
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
                                    logger.debug(
                                        f"Received control signal '{data}' for {agent_run_id}"
                                    )
                                    await message_queue.put(
                                        {"type": "control", "data": data}
                                    )
                                    return  # Stop listening on control signal

                        except StopAsyncIteration:
                            logger.warning(f"Listener stopped for {agent_run_id}.")
                            await message_queue.put(
                                {
                                    "type": "error",
                                    "data": "Listener stopped unexpectedly",
                                }
                            )
                            return
                        except Exception as e:
                            logger.error(f"Error in listener for {agent_run_id}: {e}")
                            await message_queue.put(
                                {"type": "error", "data": "Listener failed"}
                            )
                            return
                        finally:
                            # Resubscribe to the next message if continuing
                            if not terminate_stream:
                                task = asyncio.create_task(listener.__anext__())

            listener_task = asyncio.create_task(listen_messages())

            # 4. Main loop to process messages from the queue
            while not terminate_stream:
                try:
                    queue_item = await message_queue.get()

                    if queue_item["type"] == "new_response":
                        # Fetch new responses from Redis list starting after the last processed index
                        new_start_index = last_processed_index + 1
                        new_responses_json = await redis.lrange(
                            response_list_key, new_start_index, -1
                        )

                        if new_responses_json:
                            new_responses = [json.loads(r) for r in new_responses_json]
                            num_new = len(new_responses)
                            # logger.debug(f"Received {num_new} new responses for {agent_run_id} (index {new_start_index} onwards)")
                            for response in new_responses:
                                yield f"data: {json.dumps(response)}\n\n"
                                # Check if this response signals completion
                                if response.get("type") == "status" and response.get(
                                    "status"
                                ) in ["completed", "failed", "stopped"]:
                                    logger.debug(
                                        f"Detected run completion via status message in stream: {response.get('status')}"
                                    )
                                    terminate_stream = True
                                    break  # Stop processing further new responses
                            last_processed_index += num_new
                        if terminate_stream:
                            break

                    elif queue_item["type"] == "control":
                        control_signal = queue_item["data"]
                        terminate_stream = True  # Stop the stream on any control signal
                        yield f"data: {json.dumps({'type': 'status', 'status': control_signal})}\n\n"
                        break

                    elif queue_item["type"] == "error":
                        logger.error(
                            f"Listener error for {agent_run_id}: {queue_item['data']}"
                        )
                        terminate_stream = True
                        yield f"data: {json.dumps({'type': 'status', 'status': 'error'})}\n\n"
                        break

                except asyncio.CancelledError:
                    logger.debug(
                        f"Stream generator main loop cancelled for {agent_run_id}"
                    )
                    terminate_stream = True
                    break
                except Exception as loop_err:
                    logger.error(
                        f"Error in stream generator main loop for {agent_run_id}: {loop_err}",
                        exc_info=True,
                    )
                    terminate_stream = True
                    yield f"data: {json.dumps({'type': 'status', 'status': 'error', 'message': f'Stream failed: {loop_err}'})}\n\n"
                    break

        except Exception as e:
            logger.error(
                f"Error setting up stream for agent run {agent_run_id}: {e}",
                exc_info=True,
            )
            # Only yield error if initial yield didn't happen
            if not initial_yield_complete:
                yield f"data: {json.dumps({'type': 'status', 'status': 'error', 'message': f'Failed to start stream: {e}'})}\n\n"
        finally:
            terminate_stream = True
            # Graceful shutdown order: unsubscribe → close → cancel
            try:
                if "pubsub" in locals() and pubsub:
                    await pubsub.unsubscribe(response_channel, control_channel)
                    await pubsub.close()
            except Exception as e:
                logger.debug(f"Error during pubsub cleanup for {agent_run_id}: {e}")

            if listener_task:
                listener_task.cancel()
                try:
                    await listener_task  # Reap inner tasks & swallow their errors
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.debug(f"listener_task ended with: {e}")
            # Wait briefly for tasks to cancel
            await asyncio.sleep(0.1)
            logger.debug(f"Streaming cleanup complete for agent run: {agent_run_id}")

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
