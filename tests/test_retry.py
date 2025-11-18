import pytest
from tenacity import AsyncRetrying, stop_after_attempt, wait_fixed


@pytest.mark.asyncio
async def test_retry_with_tenacity():
    async def my_func(x):
        print("\n")
        print("运行中:", x)
        raise Exception("失败，触发重试")

    retry = AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
    )

    # 正确用法
    # await retry(my_func, 123)

    async for attempt in retry:
        with attempt:
            await my_func(456)
