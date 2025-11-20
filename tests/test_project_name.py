import uuid
import pytest

from core.utils.project_helpers import generate_and_update_project_name


@pytest.mark.asyncio
async def test_generate_and_update_project_name():
    await generate_and_update_project_name(str(uuid.uuid4()), "今天厦门的天气怎么样?")
