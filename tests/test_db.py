import pytest
from sqlalchemy import text

from core.services.db import get_db


@pytest.mark.asyncio
async def test_db():
    with get_db() as db:
        result = db.execute(text("SELECT 1")).first()
        print(result)
