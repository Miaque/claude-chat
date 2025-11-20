import uuid
from datetime import datetime
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel
from sqlalchemy import TEXT, UUID, Boolean, Column, DateTime
from sqlalchemy.dialects.postgresql import JSONB

from core.services.db import Base


class Message(Base):
    __tablename__ = "messages"

    message_id = Column(UUID(as_uuid=True), primary_key=True, index=True, default=uuid4)
    thread_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    type = Column(TEXT, nullable=False)
    is_llm_message = Column(Boolean, nullable=False, default=True)
    content = Column(JSONB, nullable=False, default={})
    meta = Column(JSONB, default={})
    created_at = Column(DateTime, nullable=False, index=True)
    updated_at = Column(DateTime, nullable=False)
    agent_id = Column(UUID(as_uuid=True), index=True)
    agent_version_id = Column(UUID(as_uuid=True), index=True)


class MessageModel(BaseModel):
    message_id: uuid.UUID
    thread_id: uuid.UUID
    type: str
    is_llm_message: bool
    content: dict
    meta: dict
    created_at: datetime
    updated_at: datetime
    agent_id: Optional[uuid.UUID] = None
    agent_version_id: Optional[uuid.UUID] = None

    class Config:
        from_attributes = True
