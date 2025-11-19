from uuid import uuid4

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
