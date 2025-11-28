from datetime import datetime
from typing import Literal, Optional
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, ConfigDict
from sqlalchemy import TEXT, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from core.services.db import Base, get_db
from models.types import StringUUID


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(
        StringUUID, primary_key=True, index=True, default=lambda: str(uuid4())
    )
    thread_id: Mapped[str] = mapped_column(StringUUID, nullable=False, index=True)
    status: Mapped[str] = mapped_column(TEXT, nullable=False, default="running")
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(TEXT, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.current_timestamp(),
    )
    agent_id: Mapped[Optional[str]] = mapped_column(StringUUID, nullable=True)
    agent_version_id: Mapped[Optional[str]] = mapped_column(StringUUID, nullable=True)
    meta: Mapped[dict] = mapped_column(JSONB, nullable=True, default={})


class AgentRunModel(BaseModel):
    id: str
    thread_id: str
    status: Literal["running", "completed", "failed"]
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    agent_id: Optional[str] = None
    agent_version_id: Optional[str] = None
    meta: Optional[dict] = {}

    model_config = ConfigDict(
        from_attributes=True,
        extra="ignore",
        json_encoders={datetime: lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S")},
    )


class AgentRunTable:
    @staticmethod
    def get_by_id(agent_run_id: str, *fields) -> AgentRunModel | None:
        try:
            with get_db() as db:
                if fields:
                    agent_run = (
                        db.query(*fields).filter(AgentRun.id == agent_run_id).first()
                    )
                    return agent_run if agent_run else None
                else:
                    agent_run = (
                        db.query(AgentRun).filter(AgentRun.id == agent_run_id).first()
                    )
                    return (
                        AgentRunModel.model_validate(agent_run) if agent_run else None
                    )
        except Exception:
            logger.exception("根据id => [{}]查询agent运行记录失败", agent_run_id)
            raise

    @staticmethod
    def insert(agent_run: AgentRun) -> AgentRunModel:
        try:
            with get_db() as db:
                db.add(agent_run)
                db.commit()
                db.refresh(agent_run)
                return AgentRunModel.model_validate(agent_run)
        except Exception:
            logger.exception("插入agent运行记录失败")
            raise

    @staticmethod
    def get_running_agent_runs(
        *fields,
    ) -> list[AgentRunModel] | list[AgentRun] | None:
        try:
            with get_db() as db:
                if fields:
                    agent_runs = (
                        db.query(*fields).filter(AgentRun.status == "running").all()
                    )

                    return agent_runs or None
                else:
                    agent_runs = (
                        db.query(AgentRun).filter(AgentRun.status == "running").all()
                    )

                    return (
                        [
                            AgentRunModel.model_validate(agent_run)
                            for agent_run in agent_runs
                        ]
                        if agent_runs
                        else None
                    )
        except Exception:
            logger.exception("获取正在运行的agent运行记录失败")
            raise

    @staticmethod
    def update_status(
        agent_run_id: str, status: str, error: Optional[str] = None
    ) -> AgentRunModel | None:
        try:
            with get_db() as db:
                agent_run = (
                    db.query(AgentRun).filter(AgentRun.id == agent_run_id).first()
                )
                if agent_run:
                    agent_run.status = status
                    agent_run.completed_at = datetime.now()
                    if error:
                        agent_run.error = error
                    db.commit()
                    return AgentRunModel.model_validate(agent_run)
                else:
                    return None
        except Exception:
            logger.exception(f"更新agent运行记录状态失败: {agent_run_id}")
            raise


AgentRuns = AgentRunTable()
