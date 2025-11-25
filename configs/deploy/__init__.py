from pydantic import Field
from pydantic_settings import BaseSettings


class DeploymentConfig(BaseSettings):
    """
    应用程序部署的配置设置
    """

    APPLICATION_NAME: str = Field(
        description="应用程序名称",
        default="claude-chat",
    )

    DEBUG: bool = Field(
        description="启用调试模式",
        default=False,
    )

    DEPLOY_ENV: str = Field(
        description="部署环境(例如,'PRODUCTION', 'DEVELOPMENT'),默认为 PRODUCTION",
        default="PRODUCTION",
    )
