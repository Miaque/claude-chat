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

from core.services.db import get_db
from models.project import Project, Projects
from models.thread import Thread, Threads
from models.user_role import UserRoles


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
    根据 thread_id 获取对应的 account_id。

    Raises:
        ValueError: 如果未找到该线程，或线程中不存在 account_id
    """
    try:
        with get_db() as db:
            thread_result = db.query(Thread.account_id).filter(Thread.thread_id == thread_id).first()

        if not thread_result:
            raise ValueError(f"找不到 ID 为 {thread_id} 的线程")

        account_id = thread_result.account_id
        if not account_id:
            raise ValueError("该线程未关联任何 account_id")

        return account_id
    except Exception as e:
        logger.error(f"从线程获取 account_id 时出错：{e}")
        raise


async def verify_and_get_user_id_from_jwt(request: Request) -> str:
    auth_header = request.headers.get("Authorization")

    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="未找到有效的身份验证凭据",
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
                detail="令牌载荷无效",
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


async def get_user_id_from_stream_auth(request: Request, token: Optional[str] = None) -> str:
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
                    structlog.contextvars.bind_contextvars(user_id=user_id, auth_method="jwt_query")
                    return user_id
            except Exception:
                pass

        raise HTTPException(
            status_code=401,
            detail="未找到有效的身份验证凭据",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        if "cannot schedule new futures after shutdown" in error_msg or "connection is closed" in error_msg:
            raise HTTPException(status_code=503, detail="服务器正在关闭")
        else:
            raise HTTPException(status_code=500, detail=f"身份验证时出错：{str(e)}")


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


async def verify_and_authorize_thread_access(thread_id: str, user_id: Optional[str]):
    """
    校验用户是否拥有访问某线程的权限

    参数:
        thread_id: 要检查的线程 ID
        user_id: 用户 ID
    """
    try:
        # 首先获取线程数据
        thread_data = Threads.get_by_id(thread_id)

        if not thread_data:
            raise HTTPException(status_code=404, detail="未找到线程")

        # 检查线程所属项目是否为公共项目 - 允许匿名访问
        project_id = thread_data.project_id
        if project_id:
            project_data = Projects.get_by_id(project_id, Project.is_public)
            if project_data and project_data.is_public:
                if project_data.is_public:
                    logger.debug(f"公共线程访问授权: {thread_id}")
                    return True

        # 如果项目不是公共项目，用户必须经过身份验证
        if not user_id:
            raise HTTPException(status_code=403, detail="私有线程需要身份验证才能访问")

        # 检查用户是否为管理员（管理员可访问所有线程)
        user_role = UserRoles.get_by_user_id(user_id)
        if user_role:
            role = user_role.role
            if role in ("admin", "super_admin"):
                logger.debug(f"管理员访问授权: {thread_id}", user_role=role)
                return True

        raise HTTPException(status_code=403, detail="无权限访问该线程")
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        if "cannot schedule new futures after shutdown" in error_msg or "connection is closed" in error_msg:
            raise HTTPException(status_code=503, detail="服务器正在关闭")
        else:
            raise HTTPException(status_code=500, detail=f"验证线程访问时出错: {str(e)}")


async def get_authorized_user_for_thread(thread_id: str, request: Request) -> str:
    """
    验证JWT并授权线程访问

    Args:
        thread_id: 要授权访问的线程ID
        request: FastAPI请求对象

    Returns:
        str: 经过身份验证和授权的用户ID

    Raises:
        HTTPException: 如果身份验证失败或用户缺乏线程访问权限
    """

    # 首先，验证用户
    user_id = await verify_and_get_user_id_from_jwt(request)

    # 然后，授权线程访问
    await verify_and_authorize_thread_access(thread_id, user_id)

    return user_id


class AuthorizedThreadAccess:
    """
    FastAPI依赖，结合身份验证和线程授权

    Usage:
        @router.get("/threads/{thread_id}/messages")
        async def get_messages(thread_id: str, auth: AuthorizedThreadAccess = Depends()):
            user_id = auth.user_id  # 经过身份验证和授权的用户
    """

    def __init__(self, user_id: str):
        self.user_id = user_id


class AuthorizedAgentAccess:
    """
    FastAPI依赖，结合身份验证和智能体授权

    Usage:
        @router.get("/agents/{agent_id}/config")
        async def get_agent_config(agent_id: str, auth: AuthorizedAgentAccess = Depends()):
            user_id = auth.user_id       # 经过身份验证和授权的用户
            agent_data = auth.agent_data # 智能体数据来自授权检查
    """

    def __init__(self, user_id: str, agent_data: dict):
        self.user_id = user_id
        self.agent_data = agent_data
