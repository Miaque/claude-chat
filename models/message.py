from datetime import datetime
from typing import Optional
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, ConfigDict
from sqlalchemy import TEXT, Boolean, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from core.services.db import Base, get_db
from models.types import StringUUID


class Message(Base):
    __tablename__ = "messages"

    message_id: Mapped[str] = mapped_column(StringUUID, primary_key=True, index=True, default=lambda: str(uuid4()))
    thread_id: Mapped[str] = mapped_column(StringUUID, nullable=False, index=True)
    type: Mapped[str] = mapped_column(TEXT, nullable=False)
    is_llm_message: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    content: Mapped[dict] = mapped_column(JSONB, nullable=False, default={})
    meta: Mapped[dict] = mapped_column(JSONB, default={})
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp(), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.current_timestamp())
    agent_id: Mapped[Optional[str]] = mapped_column(StringUUID, index=True)
    agent_version_id: Mapped[Optional[str]] = mapped_column(StringUUID, index=True)


class MessageModel(BaseModel):
    message_id: str
    thread_id: str
    type: str
    is_llm_message: bool
    content: dict
    meta: dict
    created_at: datetime
    updated_at: datetime
    agent_id: Optional[str] = None
    agent_version_id: Optional[str] = None

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

    @staticmethod
    def get_latest_user_message(thread_id: str) -> MessageModel | None:
        try:
            with get_db() as db:
                message = (
                    db.query(Message)
                    .filter(Message.thread_id == thread_id)
                    .filter(Message.type == "user")
                    .order_by(Message.created_at.desc())
                    .first()
                )
                return MessageModel.model_validate(message) if message else None
        except Exception:
            logger.exception("获取最新的用户消息失败")
            raise


Messages = MessageTable()
