from datetime import datetime
from typing import Optional
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, ConfigDict
from sqlalchemy import ARRAY, TEXT, VARCHAR, Boolean, DateTime, Integer, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from core.services.db import Base, get_db
from models.types import StringUUID


class Agent(Base):
    __tablename__ = "agents"

    agent_id: Mapped[str] = mapped_column(StringUUID, primary_key=True, index=True, default=lambda: str(uuid4()))
    account_id: Mapped[str] = mapped_column(StringUUID, nullable=False, index=True)
    name: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(TEXT)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, index=True, server_default=func.current_timestamp()
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, server_default=func.current_timestamp())
    current_version_id: Mapped[Optional[str]] = mapped_column(StringUUID, index=True)
    version_count: Mapped[int] = mapped_column(Integer, default=1)
    is_public: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    tags: Mapped[Optional[list[str]]] = mapped_column(ARRAY(String), default=[], index=True)
    meta: Mapped[Optional[dict]] = mapped_column(JSONB, default={}, index=True)
    icon_name: Mapped[str] = mapped_column(VARCHAR(100), nullable=False, index=True)
    icon_color: Mapped[str] = mapped_column(VARCHAR(7), nullable=False, default="#000000")
    icon_background: Mapped[str] = mapped_column(VARCHAR(7), nullable=False, default="#F3F4F6")

    __table_args__ = UniqueConstraint(account_id, is_default, name="uix_agents_account_id_is_default")


class AgentModel(BaseModel):
    agent_id: str
    account_id: str
    name: str
    description: Optional[str] = None
    is_default: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    current_version_id: Optional[str] = None
    version_count: int = 1
    is_public: bool = False
    tags: Optional[list[str]] = []
    meta: Optional[dict] = {}
    icon_name: str
    icon_color: str = "#000000"
    icon_background: str = "#F3F4F6"

    model_config = ConfigDict(
        from_attributes=True,
        extra="ignore",
        json_encoders={datetime: lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S")},
    )


class AgentTable:
    @staticmethod
    def get_by_id(agent_id: str, *fields) -> AgentModel | Agent | None:
        try:
            with get_db() as db:
                if fields:
                    agent = db.query(*fields).filter(Agent.agent_id == agent_id).first()
                    return agent or None
                else:
                    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
                    return AgentModel.model_validate(agent) if agent else None
        except Exception:
            logger.exception("根据id => [{}]查询agent失败", agent_id)
            raise


Agents = AgentTable()
