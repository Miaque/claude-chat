import uuid
from datetime import datetime
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict
from sqlalchemy import TEXT, UUID, Column, DateTime
from sqlalchemy.dialects.postgresql import JSONB

from core.services.db import Base


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
