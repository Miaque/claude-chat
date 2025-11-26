from pydantic import BaseModel


class UnifiedAgentStartResponse(BaseModel):
    """统一代理启动响应模型（新线程和现有线程）"""

    thread_id: str
    agent_run_id: str
    status: str = "running"
