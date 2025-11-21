import os
import time
import uuid
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import APIRouter, FastAPI, Request
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


@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    structlog.contextvars.clear_contextvars()

    request_id = str(uuid.uuid4())
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    path = request.url.path
    query_params = str(request.query_params)

    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        client_ip=client_ip,
        method=method,
        path=path,
        query_params=query_params,
    )

    # 记录请求开始
    logger.debug(
        f"请求开始: {method} {path} 来自 {client_ip} | 查询参数: {query_params}"
    )

    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.debug(
            f"请求完成: {method} {path} | 状态码: {response.status_code} | 耗时: {process_time:.2f}秒"
        )
        return response
    except Exception as e:
        process_time = time.time() - start_time
        try:
            error_str = str(e)
        except Exception:
            error_str = f"错误类型 {type(e).__name__}"
        logger.error(
            f"请求失败: {method} {path} | 错误: {error_str} | 耗时: {process_time:.2f}秒"
        )
        raise


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
