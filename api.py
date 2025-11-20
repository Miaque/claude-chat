import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import APIRouter, FastAPI
from loguru import logger

from agent_runs import router as agent_runs_router
from core_utils import cleanup, initialize

os.environ["TZ"] = "Asia/Shanghai"
instance_id = "single"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.debug(f"启动 FastAPI 应用，实例 ID: {instance_id}，本地模式")
    try:
        initialize(instance_id)

        # 初始化 Redis 连接
        from core.services import redis

        try:
            await redis.initialize_async()
            logger.debug("Redis 连接初始化成功")
        except Exception as e:
            logger.error(f"Redis 连接初始化失败: {e}")

        yield

        await cleanup()

        try:
            logger.debug("关闭 Redis 连接")
            await redis.close()
            logger.debug("Redis 连接关闭成功")
        except Exception as e:
            logger.error(f"关闭 Redis 连接时出错: {e}")

    except Exception as e:
        logger.error(f"应用启动期间出错: {e}")
        raise


app = FastAPI(lifespan=lifespan)

api_router = APIRouter()

api_router.include_router(agent_runs_router)

app.include_router(api_router, prefix="/api")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
