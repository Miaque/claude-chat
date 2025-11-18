from pydantic_settings import SettingsConfigDict

from configs.middleware import MiddlewareConfig


class AppConfig(MiddlewareConfig):
    model_config = SettingsConfigDict(
        # read from dotenv format config file
        env_file=".env",
        env_file_encoding="utf-8",
        # ignore extra attributes
        extra="ignore",
    )
