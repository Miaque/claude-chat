import uuid
from datetime import datetime
from typing import Optional
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, ConfigDict
from sqlalchemy import TEXT, UUID, Boolean, Column, DateTime
from sqlalchemy.dialects.postgresql import JSONB

from core.services.db import Base, get_db


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

    model_config = ConfigDict(
        from_attributes=True,
        extra="ignore",
        json_encoders={datetime: lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S")},
    )


class MessageTable:
    @staticmethod
    def save(message: Message):
        try:
            with get_db() as db:
                db.add(message)
                db.commit()
        except Exception:
            logger.exception("保存用户消息失败")
            raise

    @staticmethod
    def insert(message: Message) -> MessageModel:
        try:
            with get_db() as db:
                db.add(message)
                db.commit()
                db.refresh(message)
                return MessageModel.model_validate(message)
        except Exception:
            logger.exception("插入用户消息失败")
            raise


Messages = MessageTable()
