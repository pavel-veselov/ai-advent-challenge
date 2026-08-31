"""Настройки приложения из переменных окружения."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str = ""
    openai_model: str = "deepseek-v4-flash"
    openai_base_url: str | None = None

    # Параметры по умолчанию для запросов
    default_temperature: float = 0.7
    default_max_tokens: int = 1024


settings = Settings()
