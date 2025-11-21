import uuid
from datetime import datetime
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict
from sqlalchemy import UUID, Boolean, Column, DateTime
from sqlalchemy.dialects.postgresql import JSONB

from core.services.db import Base


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
