from uuid import uuid4

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
    metadata = Column(JSONB, nullable=True, default={}, postgresql_using="gin")
