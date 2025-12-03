import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from time import time
from typing import Any, Optional

from loguru import logger
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client


class MCPException(Exception):
    pass


class MCPConnectionError(MCPException):
    pass


class MCPToolNotFoundError(MCPException):
    pass


class MCPToolExecutionError(MCPException):
    pass


class MCPProviderError(MCPException):
    pass


class MCPConfigurationError(MCPException):
    pass


class MCPAuthenticationError(MCPException):
    pass


class CustomMCPError(MCPException):
    pass


@dataclass(frozen=True)
class MCPConnection:
    qualified_name: str
    name: str
    config: dict[str, Any]
    enabled_tools: list[str]
    provider: str = "custom"
    external_user_id: Optional[str] = None
    session: Optional[ClientSession] = field(default=None, compare=False)
    tools: Optional[list[Any]] = field(default=None, compare=False)


@dataclass(frozen=True)
class ToolInfo:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class CustomMCPConnectionResult:
    success: bool
    qualified_name: str
    display_name: str
    tools: list[dict[str, Any]]
    config: dict[str, Any]
    url: str
    message: str


@dataclass
class MCPConnectionRequest:
    qualified_name: str
    name: str
    config: dict[str, Any]
    enabled_tools: list[str]
    provider: str = "custom"
    external_user_id: Optional[str] = None


@dataclass
class ToolExecutionRequest:
    tool_name: str
    arguments: dict[str, Any]
    external_user_id: Optional[str] = None


@dataclass
class ToolExecutionResult:
    success: bool
    result: Any
    error: Optional[str] = None


class MCPService:
    def __init__(self):
        self._logger = logger
        self._connections: OrderedDict[str, tuple[MCPConnection, float]] = OrderedDict()
        self._max_connections = 100  # 最大连接数
        self._connection_ttl = 3600  # 连接缓存 1 h

    async def connect_server(self, mcp_config: dict[str, Any], external_user_id: Optional[str] = None) -> MCPConnection:
        # 根据 type 字段确定提供商
        provider = mcp_config.get("type", mcp_config.get("provider", "custom"))

        request = MCPConnectionRequest(
            qualified_name=mcp_config.get("qualifiedName", mcp_config.get("name", "")),
            name=mcp_config.get("name", ""),
            config=mcp_config.get("config", {}),
            enabled_tools=mcp_config.get("enabledTools", mcp_config.get("enabled_tools", [])),
            provider=provider,  # 供应商
            external_user_id=external_user_id,
        )
        return await self._connect_server_internal(request)

    async def _connect_server_internal(self, request: MCPConnectionRequest) -> MCPConnection:
        logger.debug(f"正在连接 MCP 服务器：{request.qualified_name}")

        try:
            server_url = await self._get_server_url(request.qualified_name, request.config, request.provider)
            headers = self._get_headers(
                request.qualified_name, request.config, request.provider, request.external_user_id
            )

            logger.debug(f"MCP 连接详情 - 提供商: {request.provider}, 地址: {server_url}, 请求头: {headers}")

            # 添加超时防止长时间等待
            async with asyncio.timeout(30):
                async with streamablehttp_client(server_url, headers=headers) as (read_stream, write_stream, _):
                    session = ClientSession(read_stream, write_stream)
                    await session.initialize()

                    tool_result = await session.list_tools()
                    tools = tool_result.tools if tool_result else []

                    connection = MCPConnection(
                        qualified_name=request.qualified_name,
                        name=request.name,
                        config=request.config,
                        enabled_tools=request.enabled_tools,
                        provider=request.provider,
                        external_user_id=request.external_user_id,
                        session=session,
                        tools=tools,
                    )

                    # 存储带有时间戳的连接以跟踪 TTL
                    self._connections[request.qualified_name] = (connection, time())
                    # 移动到最后（最近使用）
                    self._connections.move_to_end(request.qualified_name)
                    logger.debug(f"已连接 {request.qualified_name}，共 {len(tools)} 个工具可用")

                    # 清理旧连接
                    await self._cleanup_old_connections()

                    return connection

        except TimeoutError:
            error_msg = f"{request.qualified_name} 连接 30 秒超时"
            logger.error(error_msg)
            raise MCPConnectionError(error_msg)
        except Exception as e:
            logger.error("连接 {} 失败：{}", request.qualified_name, e)
            raise MCPConnectionError("连接 MCP 服务器失败：{}", e)

    async def connect_all(self, mcp_configs: list[dict[str, Any]]) -> None:
        requests = []
        for config in mcp_configs:
            # 根据 type 字段确定提供商
            provider = config.get("type", config.get("provider", "custom"))

            request = MCPConnectionRequest(
                qualified_name=config.get("qualifiedName", config.get("name", "")),
                name=config.get("name", ""),
                config=config.get("config", {}),
                enabled_tools=config.get("enabledTools", config.get("enabled_tools", [])),
                provider=provider,  # 供应商
                external_user_id=config.get("external_user_id"),
            )
            requests.append(request)

        for request in requests:
            try:
                await self._connect_server_internal(request)
            except MCPConnectionError as e:
                logger.error("连接 {} 失败：{}", request.qualified_name, e)
                continue

    async def _cleanup_old_connections(self) -> None:
        """清理过期的连接"""
        now = time()
        expired_names = []

        # 找到过期的连接
        for name, (conn, created_at) in self._connections.items():
            if now - created_at > self._connection_ttl:
                expired_names.append(name)

        # 清理过期的连接
        for name in expired_names:
            await self.disconnect_server(name)

        while len(self._connections) > self._max_connections:
            oldest_name = next(iter(self._connections))
            await self.disconnect_server(oldest_name)

    async def disconnect_server(self, qualified_name: str) -> None:
        connection_data = self._connections.get(qualified_name)
        if connection_data:
            connection, _ = connection_data
            if connection and connection.session:
                try:
                    await connection.session.close()
                    logger.debug(f"已与 {qualified_name} 断开连接")
                except Exception as e:
                    logger.warning("断开 {} 时出错：{}", qualified_name, e)

        self._connections.pop(qualified_name, None)

    async def disconnect_all(self) -> None:
        for qualified_name in list(self._connections.keys()):
            await self.disconnect_server(qualified_name)
        self._connections.clear()
        logger.debug("已与所有 MCP 服务器断开连接")

    def get_connection(self, qualified_name: str) -> Optional[MCPConnection]:
        """获取连接，并且标记为最近使用"""
        if qualified_name in self._connections:
            connection_data = self._connections[qualified_name]
            self._connections.move_to_end(qualified_name)  # 标记为最近使用
            return connection_data[0]
        return None

    def get_all_connections(self) -> list[MCPConnection]:
        """获取所有连接（不带时间戳）"""
        return [conn for conn, _ in self._connections.values()]

    def get_all_tools_openapi(self) -> list[dict[str, Any]]:
        tools = []

        for connection in self.get_all_connections():
            if not connection.tools:
                continue

            for tool in connection.tools:
                if tool.name not in connection.enabled_tools:
                    continue

                openapi_tool = {
                    "type": "function",
                    "function": {"name": tool.name, "description": tool.description, "parameters": tool.inputSchema},
                }
                tools.append(openapi_tool)

        return tools

    async def execute_tool(
        self, tool_name: str, arguments: dict[str, Any], external_user_id: Optional[str] = None
    ) -> ToolExecutionResult:
        request = ToolExecutionRequest(tool_name=tool_name, arguments=arguments, external_user_id=external_user_id)
        return await self._execute_tool_internal(request)

    async def _execute_tool_internal(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        logger.debug(f"正在执行工具：{request.tool_name}")

        connection = self._find_tool_connection(request.tool_name)
        if not connection:
            raise MCPToolNotFoundError(f"找不到工具：{request.tool_name}")

        if not connection.session:
            raise MCPToolExecutionError(f"工具 {request.tool_name} 没有活跃会话")

        if request.tool_name not in connection.enabled_tools:
            raise MCPToolExecutionError(f"工具 {request.tool_name} 未启用")

        try:
            result = await connection.session.call_tool(request.tool_name, request.arguments)

            logger.debug(f"工具 {request.tool_name} 执行成功")

            if hasattr(result, "content"):
                content = result.content
                if isinstance(content, list) and content:
                    if hasattr(content[0], "text"):
                        result_data = content[0].text
                    else:
                        result_data = str(content[0])
                else:
                    result_data = str(content)
            else:
                result_data = str(result)

            return ToolExecutionResult(success=True, result=result_data)

        except Exception as e:
            error_msg = f"工具执行失败：{str(e)}"
            logger.error(error_msg)

            return ToolExecutionResult(success=False, result=None, error=error_msg)

    def _find_tool_connection(self, tool_name: str) -> Optional[MCPConnection]:
        for connection in self.get_all_connections():
            if not connection.tools:
                continue

            for tool in connection.tools:
                if tool.name == tool_name:
                    return connection

        return None

    async def discover_custom_tools(self, request_type: str, config: dict[str, Any]) -> CustomMCPConnectionResult:
        if request_type == "http":
            return await self._discover_http_tools(config)
        elif request_type == "sse":
            return await self._discover_sse_tools(config)
        else:
            raise CustomMCPError(f"不支持的请求类型：{request_type}")

    async def _discover_http_tools(self, config: dict[str, Any]) -> CustomMCPConnectionResult:
        url = config.get("url")
        if not url:
            raise CustomMCPError("建立 HTTP MCP 连接必须提供 URL")

        try:
            async with streamablehttp_client(url) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tool_result = await session.list_tools()

                    tools_info = []
                    for tool in tool_result.tools:
                        tools_info.append(
                            {"name": tool.name, "description": tool.description, "inputSchema": tool.inputSchema}
                        )

                    return CustomMCPConnectionResult(
                        success=True,
                        qualified_name=f"custom_http_{url.split('/')[-1]}",
                        display_name=f"自定义 HTTP MCP ({url})",
                        tools=tools_info,
                        config=config,
                        url=url,
                        message=f"已通过 HTTP 连接（共 {len(tools_info)} 个工具）",
                    )

        except Exception as e:
            logger.error(f"连接 HTTP MCP 服务器出错：{str(e)}")
            return CustomMCPConnectionResult(
                success=False,
                qualified_name="",
                display_name="",
                tools=[],
                config=config,
                url=url,
                message=f"连接失败：{str(e)}",
            )

    async def _discover_sse_tools(self, config: dict[str, Any]) -> CustomMCPConnectionResult:
        url = config.get("url")
        if not url:
            raise CustomMCPError("建立 SSE MCP 连接必须提供 URL")

        try:
            async with sse_client(url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tool_result = await session.list_tools()

                    tools_info = []
                    for tool in tool_result.tools:
                        tools_info.append(
                            {"name": tool.name, "description": tool.description, "inputSchema": tool.inputSchema}
                        )

                    return CustomMCPConnectionResult(
                        success=True,
                        qualified_name=f"custom_sse_{url.split('/')[-1]}",
                        display_name=f"自定义 SSE MCP ({url})",
                        tools=tools_info,
                        config=config,
                        url=url,
                        message=f"已通过 SSE 连接（共 {len(tools_info)} 个工具）",
                    )

        except Exception as e:
            logger.error("连接 SSE MCP 服务器失败：{}", e)
            return CustomMCPConnectionResult(
                success=False,
                qualified_name="",
                display_name="",
                tools=[],
                config=config,
                url=url,
                message=f"连接失败：{str(e)}",
            )

    async def _get_server_url(self, qualified_name: str, config: dict[str, Any], provider: str) -> str:
        if provider in ["custom", "http", "sse"]:
            return await self._get_custom_server_url(qualified_name, config)
        else:
            raise MCPProviderError(f"未知供应商类型：{provider}")

    def _get_headers(
        self, qualified_name: str, config: dict[str, Any], provider: str, external_user_id: Optional[str] = None
    ) -> dict[str, str]:
        if provider in ["custom", "http", "sse"]:
            return self._get_custom_headers(qualified_name, config, external_user_id)
        else:
            raise MCPProviderError(f"未知供应商类型：{provider}")

    async def _get_custom_server_url(self, qualified_name: str, config: dict[str, Any]) -> str:
        url = config.get("url")
        if not url:
            raise MCPProviderError(f"自定义 MCP 服务器 {qualified_name} 未提供 URL")
        return url

    def _get_custom_headers(
        self, qualified_name: str, config: dict[str, Any], external_user_id: Optional[str] = None
    ) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}

        if "headers" in config:
            headers.update(config["headers"])

        if external_user_id:
            headers["X-External-User-Id"] = external_user_id

        return headers


mcp_service = MCPService()
