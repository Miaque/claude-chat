import pytest
from sqlalchemy import text

from core.services.db import get_db
from models.project import Project


@pytest.mark.asyncio
async def test_db():
    with get_db() as db:
        result = db.execute(text("SELECT 1")).first()
        print(result)


def test_project_query():
    with get_db() as db:
        result = db.query(Project).first()
        print(result)
