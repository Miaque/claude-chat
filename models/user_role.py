from datetime import datetime
from typing import Optional
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, ConfigDict
from sqlalchemy import VARCHAR, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from core.services.db import Base, get_db
from models.types import StringUUID


class UserRole(Base):
    __tablename__ = "user_roles"

    user_id: Mapped[str] = mapped_column(StringUUID, primary_key=True, index=True, default=lambda: str(uuid4()))
    role: Mapped[str] = mapped_column(VARCHAR(50), nullable=False, default="user", index=True)
    granted_by: Mapped[str] = mapped_column(StringUUID)
    granted_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    meta: Mapped[dict] = mapped_column(JSONB, default={})


class UserRoleModel(BaseModel):
    user_id: str
    role: str
    granted_by: Optional[str] = None
    granted_at: Optional[datetime] = None
    meta: dict

    model_config = ConfigDict(
        from_attributes=True,
        extra="ignore",
        json_encoders={datetime: lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S")},
    )


class UserRoleTable:
    @staticmethod
    def get_by_user_id(user_id: str) -> UserRoleModel | None:
        try:
            with get_db() as db:
                user_role = db.query(UserRole).filter(UserRole.user_id == user_id).first()
                return UserRoleModel.model_validate(user_role) if user_role else None
        except Exception:
            logger.exception("根据用户ID查询用户角色失败")
            raise


UserRoles = UserRoleTable()
