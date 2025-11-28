from datetime import datetime
from typing import Optional
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, ConfigDict
from sqlalchemy import Boolean, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from core.services.db import Base, get_db
from models.types import StringUUID


class Thread(Base):
    __tablename__ = "threads"

    thread_id: Mapped[str] = mapped_column(
        StringUUID, primary_key=True, index=True, default=lambda: str(uuid4())
    )
    account_id: Mapped[str] = mapped_column(StringUUID, index=True)
    project_id: Mapped[str] = mapped_column(StringUUID, index=True)
    is_public: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp(), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )
    meta: Mapped[dict] = mapped_column(JSONB, nullable=True, default={})
    session_id: Mapped[Optional[str]] = mapped_column(
        StringUUID, nullable=True, index=True
    )


class ThreadModel(BaseModel):
    thread_id: str
    account_id: Optional[str] = None
    project_id: Optional[str] = None
    is_public: Optional[bool] = False
    created_at: datetime
    updated_at: datetime
    meta: Optional[dict] = {}
    session_id: Optional[str] = None

    model_config = ConfigDict(
        from_attributes=True,
        extra="ignore",
        json_encoders={datetime: lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S")},
    )


class ThreadTable:
    @staticmethod
    def insert(thread: Thread) -> ThreadModel:
        try:
            with get_db() as db:
                db.add(thread)
                db.commit()
                db.refresh(thread)
                return ThreadModel.model_validate(thread)
        except Exception:
            logger.exception("创建新线程失败")
            raise

    @staticmethod
    def get_by_id(thread_id: str, *fields) -> ThreadModel | None | Thread:
        try:
            with get_db() as db:
                # 如果指定了 fields，只查询指定字段
                if fields:
                    # 执行字段级查询
                    result = (
                        db.query(*fields).filter(Thread.thread_id == thread_id).first()
                    )
                    return result or None

                # 没有指定 fields，查询完整对象
                else:
                    thread = (
                        db.query(Thread).filter(Thread.thread_id == thread_id).first()
                    )
                    return ThreadModel.model_validate(thread) if thread else None

        except Exception:
            logger.exception(f"根据id => [{thread_id}]查询线程失败")
            return None

    @staticmethod
    def get_by_ids(thread_ids: list[str], account_id: str) -> list[ThreadModel] | None:
        try:
            with get_db() as db:
                threads = (
                    db.query(Thread)
                    .filter(Thread.thread_id.in_(thread_ids))
                    .filter(Thread.account_id == account_id)
                    .all()
                )
                return (
                    [ThreadModel.model_validate(thread) for thread in threads]
                    if threads
                    else None
                )
        except Exception:
            logger.exception(
                f"根据ids => [{thread_ids}]和account_id => [{account_id}]查询线程失败"
            )
            raise

    @staticmethod
    def update_session_id(thread_id: str, session_id: str):
        try:
            with get_db() as db:
                db.query(Thread).filter(Thread.thread_id == thread_id).update(
                    {"session_id": session_id, "created_at": datetime.now()}
                )
                db.commit()
        except Exception:
            logger.exception(f"更新线程 {thread_id} 的 session_id 失败")
            raise


Threads = ThreadTable()
