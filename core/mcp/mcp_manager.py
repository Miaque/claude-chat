from typing import Optional

from claude_agent_sdk.types import McpHttpServerConfig, McpServerConfig, McpSSEServerConfig, McpStdioServerConfig

from core.thread_manager import ThreadManager


class MCPManager:
    def __init__(self, thread_manager: ThreadManager, account_id: str):
        self.thread_manager = thread_manager
        self.account_id = account_id

    async def get_mcp_servers(self, agent_config: dict) -> Optional[dict[str, McpServerConfig]]:
        all_mcps = []

        if agent_config.get("configured_mcps"):
            all_mcps.extend(agent_config["configured_mcps"])

        if not all_mcps:
            return None

        return {mcp["name"]: self._get_mcp_server_config(mcp) for mcp in all_mcps}

    def _get_mcp_server_config(self, mcp_config: dict) -> McpServerConfig:
        if mcp_config.get("type") == "sse":
            return McpSSEServerConfig(
                type="sse",
                url=mcp_config["url"],
                headers=mcp_config.get("headers", {}),
            )
        elif mcp_config.get("type") == "http":
            return McpHttpServerConfig(
                type="http",
                url=mcp_config["url"],
                headers=mcp_config.get("headers", {}),
            )
        elif mcp_config.get("type") == "stdio":
            return McpStdioServerConfig(
                type="stdio",
                command=mcp_config["command"],
                args=mcp_config.get("args", []),
                env=mcp_config.get("env", {}),
            )
        else:
            raise ValueError(f"不支持的MCP类型: {mcp_config.get('type')}")
