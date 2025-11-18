# CLAUDE.md

本文档为 Claude Code (claude.ai/code) 在与本仓库代码协作时提供指导。

## 项目概述

这是一个基于 **claude-chat** 的 Python 应用程序，使用 FastAPI 构建 Web API，Dramatiq 处理后台任务。项目使用 Redis 作为消息代理和缓存层。

## 开发命令

### 运行应用程序

- **开发服务器**: `uvicorn api:app --host 0.0.0.0 --port 8080 --reload`
- **生产服务器**: `uvicorn api:app --host 0.0.0.0 --port 8080`
- **VS Code 调试**: 使用 `.vscode/launch.json` 中的 "api" 启动配置

### 测试

- **运行测试**: `pytest -s` (在 `pytest.ini` 中配置，支持 `.env` 文件)
- **测试配置**: 测试使用 pytest，并从 `.env` 加载环境变量

### 代码质量

- **代码检查**: `uv run ruff check` (在 `pyproject.toml` 中配置)
- **代码格式化**: `uv run ruff format`

### 依赖管理

- **包管理器**: uv (根据 `uv.lock` 文件判断)
- **安装依赖**: `uv sync` 或 `uv pip install -r pyproject.toml`

## 架构

### 核心组件

1. **FastAPI Web 层** (`api.py`)

   - HTTP API 的主要入口
   - 目前提供基础端点: `/` 和 `/hello/{name}`

2. **后台任务处理** (`run_agent_background.py`)

   - 使用 Redis 代理的 Dramatiq 工作进程配置
   - 带重试逻辑的异步初始化模式
   - 使用 `tenacity` 实现重试机制

3. **配置系统** (`configs/`)

   - 使用 Pydantic Settings 的分层配置
   - `AppConfig` 继承 `MiddlewareConfig`，后者组合了 `DatabaseConfig` 和 `RedisConfig`
   - 通过 `.env` 文件实现基于环境的配置
   - 全面的 Redis 配置，支持集群、哨兵模式和 SSL

4. **核心服务** (`core/`)
   - `core/services/redis.py`: 带连接池的异步 Redis 客户端
   - `core/utils/retry.py`: 通用的异步重试工具

### 关键设计模式

1. **异步优先**: 所有 Redis 操作和后台任务都使用 async/await
2. **配置驱动**: 广泛使用 Pydantic 实现类型安全的配置
3. **重试逻辑**: 关键操作内置重试机制 (Redis 连接、后台任务)
4. **连接池**: Redis 连接使用连接池以提高性能

### 依赖项

- **FastAPI**: Web 框架
- **Dramatiq**: 使用 Redis 代理的后台任务处理
- **Redis**: 消息代理和缓存
- **Pydantic**: 配置管理和数据验证
- **Loguru**: 结构化日志
- **Tenacity**: 重试机制
- **Uvicorn**: ASGI 服务器

### 环境配置

应用程序期望 `.env` 文件包含以下配置:

- Redis 连接 (主机、端口、密码等)
- 数据库连接 (PostgreSQL)
- 应用程序特定设置

所有配置都通过 `AppConfig` 类处理，并带有合理的默认值。
