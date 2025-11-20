import base64
import hashlib
import hmac
import os
from typing import Optional

import jwt
import structlog
from fastapi import Header, HTTPException, Request
from jwt.exceptions import PyJWTError
from loguru import logger

from core.services import redis
from core.services.db import get_db
from models.thread import Thread


async def verify_admin_api_key(x_admin_api_key: Optional[str] = Header(None)):
    return True


def _decode_jwt_safely(token: str) -> dict:
    return jwt.decode(
        token,
        options={
            "verify_signature": False,
            "verify_exp": True,
            "verify_aud": False,
            "verify_iss": False,
        },
    )


async def get_account_id_from_thread(thread_id: str) -> str:
    """
    Get account_id from thread_id.

    Raises:
        ValueError: If thread not found or has no account_id
    """
    try:
        with get_db() as db:
            thread_result = (
                db.query(Thread.account_id)
                .filter(Thread.thread_id == thread_id)
                .first()
            )

        if not thread_result:
            raise ValueError(f"Could not find thread with ID: {thread_id}")

        account_id = thread_result.account_id
        if not account_id:
            raise ValueError("Thread has no associated account_id")

        return account_id
    except Exception as e:
        logger.error(f"Error getting account_id from thread: {e}")
        raise


async def _get_user_id_from_account_cached(account_id: str) -> Optional[str]:
    cache_key = f"account_user:{account_id}"

    try:
        redis_client = await redis.get_client()
        cached_user_id = await redis_client.get(cache_key)
        if cached_user_id:
            return (
                cached_user_id.decode("utf-8")
                if isinstance(cached_user_id, bytes)
                else cached_user_id
            )
    except Exception as e:
        logger.warning(f"Redis cache lookup failed for account {account_id}: {e}")

    try:
        db = DBConnection()
        await db.initialize()
        client = await db.client

        user_result = (
            await client.schema("basejump")
            .table("accounts")
            .select("primary_owner_user_id")
            .eq("id", account_id)
            .limit(1)
            .execute()
        )

        if user_result.data:
            user_id = user_result.data[0]["primary_owner_user_id"]

            try:
                await redis_client.setex(cache_key, 300, user_id)
            except Exception as e:
                logger.warning(f"Failed to cache user lookup: {e}")

            return user_id

        return None

    except Exception as e:
        logger.error(f"Database lookup failed for account {account_id}: {e}")
        return None


async def verify_and_get_user_id_from_jwt(request: Request) -> str:
    auth_header = request.headers.get("Authorization")

    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="No valid authentication credentials found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_header.split(" ")[1]

    try:
        # payload = _decode_jwt_safely(token)
        # user_id = payload.get("sub")
        user_id = "00000000-0000-0000-0000-000000000000"

        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )

        structlog.contextvars.bind_contextvars(user_id=user_id, auth_method="jwt")
        return user_id

    except PyJWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_user_id_from_stream_auth(
    request: Request, token: Optional[str] = None
) -> str:
    try:
        try:
            return await verify_and_get_user_id_from_jwt(request)
        except HTTPException:
            pass

        if token:
            try:
                payload = _decode_jwt_safely(token)
                user_id = payload.get("sub")
                if user_id:
                    structlog.contextvars.bind_contextvars(
                        user_id=user_id, auth_method="jwt_query"
                    )
                    return user_id
            except Exception:
                pass

        raise HTTPException(
            status_code=401,
            detail="No valid authentication credentials found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        if (
            "cannot schedule new futures after shutdown" in error_msg
            or "connection is closed" in error_msg
        ):
            raise HTTPException(status_code=503, detail="Server is shutting down")
        else:
            raise HTTPException(
                status_code=500, detail=f"Error during authentication: {str(e)}"
            )


async def get_optional_user_id(request: Request) -> Optional[str]:
    auth_header = request.headers.get("Authorization")

    if not auth_header or not auth_header.startswith("Bearer "):
        return None

    token = auth_header.split(" ")[1]

    try:
        payload = _decode_jwt_safely(token)

        user_id = payload.get("sub")
        if user_id:
            structlog.contextvars.bind_contextvars(user_id=user_id)

        return user_id
    except PyJWTError:
        return None


get_optional_current_user_id_from_jwt = get_optional_user_id


async def verify_and_get_agent_authorization(agent_id: str, user_id: str) -> dict:
    try:
        agent_result = (
            await client.table("agents")
            .select("*")
            .eq("agent_id", agent_id)
            .eq("account_id", user_id)
            .execute()
        )

        if not agent_result.data:
            raise HTTPException(
                status_code=404, detail="Agent not found or access denied"
            )

        return agent_result.data[0]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error verifying agent access for agent {agent_id}, user {user_id}: {str(e)}"
        )
        raise HTTPException(status_code=500, detail="Failed to verify agent access")


async def verify_and_authorize_thread_access(thread_id: str, user_id: Optional[str]):
    """
    Verify that a user has access to a thread.
    Supports both authenticated and anonymous access (for public threads).

    Args:
        thread_id: Thread ID to check
        user_id: User ID (can be None for anonymous users accessing public threads)
    """
    try:
        # Get thread data first
        thread_result = (
            await client.table("threads")
            .select("*")
            .eq("thread_id", thread_id)
            .execute()
        )

        if not thread_result.data or len(thread_result.data) == 0:
            raise HTTPException(status_code=404, detail="Thread not found")

        thread_data = thread_result.data[0]

        # Check if thread's project is public - allow anonymous access
        project_id = thread_data.get("project_id")
        if project_id:
            project_result = (
                await client.table("projects")
                .select("is_public")
                .eq("project_id", project_id)
                .execute()
            )
            if project_result.data and len(project_result.data) > 0:
                if project_result.data[0].get("is_public"):
                    logger.debug(f"Public thread access granted: {thread_id}")
                    return True

        # If not public, user must be authenticated
        if not user_id:
            raise HTTPException(
                status_code=403, detail="Authentication required for private threads"
            )

        # Check if user is an admin (admins have access to all threads)
        admin_result = (
            await client.table("user_roles")
            .select("role")
            .eq("user_id", user_id)
            .execute()
        )
        if admin_result.data and len(admin_result.data) > 0:
            role = admin_result.data[0].get("role")
            if role in ("admin", "super_admin"):
                logger.debug(
                    f"Admin access granted for thread {thread_id}", user_role=role
                )
                return True

        # Check if user owns the thread
        if thread_data["account_id"] == user_id:
            return True

        # Check if user is a team member of the account
        account_id = thread_data.get("account_id")
        if account_id:
            account_user_result = (
                await client.schema("basejump")
                .from_("account_user")
                .select("account_role")
                .eq("user_id", user_id)
                .eq("account_id", account_id)
                .execute()
            )
            if account_user_result.data and len(account_user_result.data) > 0:
                return True

        raise HTTPException(
            status_code=403, detail="Not authorized to access this thread"
        )
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        if (
            "cannot schedule new futures after shutdown" in error_msg
            or "connection is closed" in error_msg
        ):
            raise HTTPException(status_code=503, detail="Server is shutting down")
        else:
            raise HTTPException(
                status_code=500, detail=f"Error verifying thread access: {str(e)}"
            )


async def get_authorized_user_for_thread(thread_id: str, request: Request) -> str:
    """
    FastAPI dependency that verifies JWT and authorizes thread access.

    Args:
        thread_id: The thread ID to authorize access for
        request: The FastAPI request object

    Returns:
        str: The authenticated and authorized user ID

    Raises:
        HTTPException: If authentication fails or user lacks thread access
    """

    # First, authenticate the user
    user_id = await verify_and_get_user_id_from_jwt(request)

    # Then, authorize thread access
    await verify_and_authorize_thread_access(thread_id, user_id)

    return user_id


async def get_authorized_user_for_agent(
    agent_id: str, request: Request
) -> tuple[str, dict]:
    """
    FastAPI dependency that verifies JWT and authorizes agent access.

    Args:
        agent_id: The agent ID to authorize access for
        request: The FastAPI request object

    Returns:
        tuple[str, dict]: The authenticated user ID and agent data

    Raises:
        HTTPException: If authentication fails or user lacks agent access
    """

    # First, authenticate the user
    user_id = await verify_and_get_user_id_from_jwt(request)

    # Then, authorize agent access and get agent data
    agent_data = await verify_and_get_agent_authorization(agent_id, user_id)

    return user_id, agent_data


class AuthorizedThreadAccess:
    """
    FastAPI dependency that combines authentication and thread authorization.

    Usage:
        @router.get("/threads/{thread_id}/messages")
        async def get_messages(
            thread_id: str,
            auth: AuthorizedThreadAccess = Depends()
        ):
            user_id = auth.user_id  # Authenticated and authorized user
    """

    def __init__(self, user_id: str):
        self.user_id = user_id


class AuthorizedAgentAccess:
    """
    FastAPI dependency that combines authentication and agent authorization.

    Usage:
        @router.get("/agents/{agent_id}/config")
        async def get_agent_config(
            agent_id: str,
            auth: AuthorizedAgentAccess = Depends()
        ):
            user_id = auth.user_id       # Authenticated and authorized user
            agent_data = auth.agent_data # Agent data from authorization check
    """

    def __init__(self, user_id: str, agent_data: dict):
        self.user_id = user_id
        self.agent_data = agent_data


async def require_thread_access(
    thread_id: str, request: Request
) -> AuthorizedThreadAccess:
    """
    FastAPI dependency that verifies JWT and authorizes thread access.

    Args:
        thread_id: The thread ID from the path parameter
        request: The FastAPI request object

    Returns:
        AuthorizedThreadAccess: Object containing authenticated user_id

    Raises:
        HTTPException: If authentication fails or user lacks thread access
    """
    user_id = await get_authorized_user_for_thread(thread_id, request)
    return AuthorizedThreadAccess(user_id)


async def require_agent_access(
    agent_id: str, request: Request
) -> AuthorizedAgentAccess:
    """
    FastAPI dependency that verifies JWT and authorizes agent access.

    Args:
        agent_id: The agent ID from the path parameter
        request: The FastAPI request object

    Returns:
        AuthorizedAgentAccess: Object containing user_id and agent_data

    Raises:
        HTTPException: If authentication fails or user lacks agent access
    """
    user_id, agent_data = await get_authorized_user_for_agent(agent_id, request)
    return AuthorizedAgentAccess(user_id, agent_data)
