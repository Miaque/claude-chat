from typing import Any, Optional

from pydantic import BaseModel


class AgentVersionResponse(BaseModel):
    """代理版本信息响应模型"""

    version_id: str
    agent_id: str
    version_number: int
    version_name: str
    configured_mcps: list[dict[str, Any]]
    is_active: bool
    created_at: str
    updated_at: str
    created_by: Optional[str] = None


class AgentResponse(BaseModel):
    """代理信息响应模型"""

    agent_id: str
    name: str
    description: Optional[str] = None
    configured_mcps: list[dict[str, Any]]
    is_default: bool
    is_public: Optional[bool] = False
    tags: Optional[list[str]] = []
    icon_name: Optional[str] = None
    icon_color: Optional[str] = None
    icon_background: Optional[str] = None
    created_at: str
    updated_at: Optional[str] = None
    current_version_id: Optional[str] = None
    version_count: Optional[int] = 1
    current_version: Optional[AgentVersionResponse] = None
    metadata: Optional[dict[str, Any]] = None
    account_id: Optional[str] = None  # 内部字段，响应中可能不需要


class ThreadAgentResponse(BaseModel):
    """线程代理信息响应模型"""

    agent: Optional[AgentResponse]
    source: str
    message: str
