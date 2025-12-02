import pytest

from core.utils.icon_generator import generate_icon_and_colors


@pytest.mark.asyncio
async def test_generate_icon_and_colors():
    result = await generate_icon_and_colors("智能体", "今天北京的天气怎么样?")
    assert result is not None
    assert result["icon_name"] is not None
    assert result["icon_background"] is not None
    assert result["icon_color"] is not None
