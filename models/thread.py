import uuid
from datetime import datetime
from typing import Optional
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, ConfigDict
from sqlalchemy import UUID, Boolean, Column, DateTime, ColumnElement
from sqlalchemy.dialects.postgresql import JSONB

from core.services.db import Base, get_db


class Thread(Base):
    __tablename__ = "threads"

    thread_id = Column(UUID(as_uuid=True), primary_key=True, index=True, default=uuid4)
    account_id = Column(UUID(as_uuid=True), index=True)
    project_id = Column(UUID(as_uuid=True), index=True)
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime, nullable=False, index=True)
    updated_at = Column(DateTime, nullable=False)
    meta = Column(JSONB, nullable=True, default={})


class ThreadModel(BaseModel):
    thread_id: uuid.UUID
    account_id: Optional[uuid.UUID] = None
    project_id: Optional[uuid.UUID] = None
    is_public: Optional[bool] = False
    created_at: datetime
    updated_at: datetime
    meta: Optional[dict] = {}

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
    def get_by_id(
        thread_id: str, *fields: ColumnElement
    ) -> ThreadModel | None | Thread:
        try:
            with get_db() as db:
                # 如果指定了 fields，只查询指定字段
                if fields:
                    # 执行字段级查询
                    result = (
                        db.query(*fields).filter(Thread.thread_id == thread_id).first()
                    )
                    return result if result else None

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


Threads = ThreadTable()
