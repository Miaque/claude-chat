from datetime import datetime
from typing import Optional
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, ConfigDict
from sqlalchemy import Boolean, DateTime, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from core.services.db import Base, get_db
from models.types import StringUUID


class Project(Base):
    __tablename__ = "projects"

    project_id: Mapped[str] = mapped_column(
        StringUUID, primary_key=True, index=True, default=lambda: str(uuid4())
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    account_id: Mapped[str] = mapped_column(StringUUID, nullable=False, index=True)
    sandbox: Mapped[dict] = mapped_column(JSONB, default={})
    is_public: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp(), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )
    icon_name: Mapped[Optional[str]] = mapped_column(Text)


class ProjectModel(BaseModel):
    project_id: str
    name: str
    description: Optional[str] = None
    account_id: str
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


class ProjectTable:
    @staticmethod
    def insert(project: Project) -> ProjectModel:
        try:
            with get_db() as db:
                db.add(project)
                db.commit()
                db.refresh(project)
                return ProjectModel.model_validate(project)
        except Exception:
            logger.exception("创建新项目失败")
            raise

    @staticmethod
    def get_by_id(project_id: str, *fields) -> ProjectModel | Project | None:
        try:
            with get_db() as db:
                if fields:
                    response = (
                        db.query(*fields)
                        .filter(Project.project_id == project_id)
                        .first()
                    )
                    return response if response else None
                else:
                    response = (
                        db.query(Project)
                        .filter(Project.project_id == project_id)
                        .first()
                    )
                return ProjectModel.model_validate(response)
        except Exception:
            logger.exception("根据id查询项目失败")
            raise


Projects = ProjectTable()
