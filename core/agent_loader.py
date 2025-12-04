from dataclasses import dataclass
from typing import Any, Optional

from core.runtime_cache import get_cached_agent_config, set_cached_agent_config
from models.agent import AgentModel, Agents
from schemas.agents import AgentResponse, AgentVersionResponse


@dataclass
class AgentData:
    agent_id: str
    name: str
    description: Optional[str]
    account_id: str
    is_default: bool
    is_public: bool
    tags: list
    icon_name: Optional[str]
    icon_color: Optional[str]
    icon_background: Optional[str]
    created_at: str
    updated_at: str
    current_version_id: Optional[str]
    version_count: int
    metadata: Optional[dict[str, Any]]
    configured_mcps: Optional[list] = None

    # 版本信息
    version_name: Optional[str] = None
    version_number: Optional[int] = None
    version_created_at: Optional[str] = None
    version_updated_at: Optional[str] = None
    version_created_by: Optional[str] = None

    # 元数据标志
    is_global_default: bool = False
    config_loaded: bool = False

    def to_pydantic_model(self):
        current_version = None
        if self.config_loaded and self.version_number is not None:
            current_version = AgentVersionResponse(
                version_id=self.current_version_id or "",
                agent_id=self.agent_id,
                version_number=self.version_number,
                version_name=self.version_name or "v1",
                configured_mcps=self.configured_mcps or [],
                is_active=True,
                created_at=self.version_created_at or self.created_at,
                updated_at=self.version_updated_at or self.updated_at,
                created_by=self.version_created_by,
            )

        return AgentResponse(
            agent_id=self.agent_id,
            name=self.name,
            description=self.description,
            configured_mcps=self.configured_mcps or [],
            is_default=self.is_default,
            is_public=self.is_public,
            tags=self.tags,
            icon_name=self.icon_name,
            icon_color=self.icon_color,
            icon_background=self.icon_background,
            created_at=self.created_at,
            updated_at=self.updated_at,
            current_version_id=self.current_version_id,
            version_count=self.version_count,
            current_version=current_version,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "account_id": self.account_id,
            "is_default": self.is_default,
            "is_public": self.is_public,
            "tags": self.tags,
            "icon_name": self.icon_name,
            "icon_color": self.icon_color,
            "icon_background": self.icon_background,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "current_version_id": self.current_version_id,
            "version_count": self.version_count,
            "metadata": self.metadata,
        }

        if self.config_loaded:
            result.update(
                {
                    "configured_mcps": self.configured_mcps,
                    "version_name": self.version_name,
                    "is_global_default": self.is_global_default,
                }
            )

            if self.version_number is not None:
                result["current_version"] = {
                    "version_id": self.current_version_id,
                    "version_number": self.version_number,
                    "version_name": self.version_name,
                    "created_at": self.version_created_at,
                    "updated_at": self.version_updated_at,
                    "created_by": self.version_created_by,
                }
        else:
            result.update(
                {
                    "configured_mcps": [],
                }
            )

        return result


class AgentLoader:
    def __init__(self):
        pass

    async def load_agent(
        self, agent_id: str, user_id: str, load_config: bool = True, skip_cache: bool = False
    ) -> AgentData:
        if load_config and not skip_cache:
            cached = await get_cached_agent_config(agent_id)
            if cached:
                return self._dict_to_agent_data(cached)

        agent = Agents.get_by_id(agent_id)
        if not agent:
            raise ValueError(f"找不到智能体 {agent_id}")

        # 检查权限
        if agent.account_id != user_id and not agent.is_public:
            raise ValueError(f"无权限访问智能体 {agent_id}")

        agent_data = self._row_to_agent_data(agent)
        agent_data.config_loaded = True

        await set_cached_agent_config(
            agent_id,
            agent_data.to_dict(),
            version_id=agent.current_version_id,
            is_global_default=agent_data.is_global_default,
        )

        return agent_data

    def _dict_to_agent_data(self, data: dict[str, Any]) -> AgentData:
        current_version = data.get("current_version", {}) or {}

        return AgentData(
            agent_id=data["agent_id"],
            name=data["name"],
            description=data.get("description"),
            account_id=data["account_id"],
            is_default=data.get("is_default", False),
            is_public=data.get("is_public", False),
            tags=data.get("tags", []),
            icon_name=data.get("icon_name"),
            icon_color=data.get("icon_color"),
            icon_background=data.get("icon_background"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            current_version_id=data.get("current_version_id"),
            version_count=data.get("version_count", 1),
            metadata=data.get("meta", {}),
            configured_mcps=data.get("configured_mcps", []),
            version_name=data.get("version_name") or current_version.get("version_name"),
            version_number=current_version.get("version_number"),
            version_created_at=current_version.get("created_at"),
            version_updated_at=current_version.get("updated_at"),
            version_created_by=current_version.get("created_by"),
            is_global_default=data.get("is_global_default", False),
            config_loaded=True,
        )

    def _row_to_agent_data(self, row: AgentModel) -> AgentData:
        metadata = row.meta or {}
        created_at = row.created_at.strftime("%Y-%m-%d %H:%M:%S") if row.created_at else ""
        updated_at = row.updated_at.strftime("%Y-%m-%d %H:%M:%S") if row.updated_at else created_at

        return AgentData(
            agent_id=row.agent_id,
            name=row.name,
            description=row.description,
            account_id=row.account_id,
            is_default=row.is_default,
            is_public=row.is_public,
            tags=row.tags or [],
            icon_name=row.icon_name,
            icon_color=row.icon_color,
            icon_background=row.icon_background,
            created_at=created_at,
            updated_at=updated_at,
            current_version_id=row.current_version_id,
            version_count=row.version_count,
            metadata=metadata,
            configured_mcps=metadata.get("configured_mcps", []),
            is_global_default=metadata.get("is_global_default", False),
            config_loaded=False,
        )

    def _load_fallback_config(self, agent: AgentData):
        agent.configured_mcps = []
        agent.version_name = "v1"

    def _apply_version_config(self, agent: AgentData, version_row: dict[str, Any]):
        config = version_row.get("config") or {}
        tools = config.get("tools", {})

        agent.configured_mcps = tools.get("mcp", [])
        agent.version_name = version_row.get("version_name", "v1")
        agent.version_number = version_row.get("version_number")


# 单例
_loader = None


async def get_agent_loader() -> AgentLoader:
    global _loader
    if _loader is None:
        _loader = AgentLoader()
    return _loader
