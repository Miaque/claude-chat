import uuid
from datetime import datetime
from typing import List, Literal, Optional
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, ConfigDict
from sqlalchemy import TEXT, UUID, Column, ColumnElement, DateTime
from sqlalchemy.dialects.postgresql import JSONB

from core.services.db import Base, get_db


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, index=True, default=uuid4)
    thread_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    status = Column(TEXT, nullable=False, default="running")
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    error = Column(TEXT, nullable=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    agent_id = Column(UUID(as_uuid=True), nullable=True)
    agent_version_id = Column(UUID(as_uuid=True), nullable=True)
    meta = Column(JSONB, nullable=True, default={})


class AgentRunModel(BaseModel):
    id: uuid.UUID
    thread_id: uuid.UUID
    status: Literal["running", "completed", "failed"]
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    agent_id: Optional[uuid.UUID] = None
    agent_version_id: Optional[uuid.UUID] = None
    meta: Optional[dict] = {}

    model_config = ConfigDict(
        from_attributes=True,
        extra="ignore",
        json_encoders={datetime: lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S")},
    )


class AgentRunTable:
    @staticmethod
    def get_by_id(agent_run_id: str) -> AgentRunModel | None:
        try:
            with get_db() as db:
                agent_run = (
                    db.query(AgentRun).filter(AgentRun.id == agent_run_id).first()
                )
                return AgentRunModel.model_validate(agent_run) if agent_run else None
        except Exception:
            logger.exception(f"根据id => [{agent_run_id}]查询agent运行记录失败")
            return None

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
        *fields: ColumnElement,
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


AgentRuns = AgentRunTable()
