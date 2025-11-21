import uuid
from datetime import datetime
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict
from sqlalchemy import UUID, Boolean, Column, DateTime, Text
from sqlalchemy.dialects.postgresql import JSONB

from core.services.db import Base


class Project(Base):
    __tablename__ = "projects"

    project_id = Column(UUID(as_uuid=True), primary_key=True, index=True, default=uuid4)
    name = Column(Text, nullable=False)
    description = Column(Text)
    account_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    sandbox = Column(JSONB, default={})
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime, nullable=False, index=True)
    updated_at = Column(DateTime, nullable=False)
    icon_name = Column(Text)


class ProjectModel(BaseModel):
    project_id: uuid.UUID
    name: str
    description: Optional[str] = None
    account_id: uuid.UUID
    sandbox: dict
    is_public: bool
    created_at: datetime
    updated_at: datetime
    icon_name: Optional[str] = None

    model_config = ConfigDict(
        from_attributes=True,
        extra="ignore",
        json_encoders={datetime: lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S")},
    )
