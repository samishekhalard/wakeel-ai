from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str = "sk-placeholder"
    llm_model: str = "meta-llama/llama-3.1-8b-instruct:free"
    database_url: str = "sqlite+aiosqlite:///./wakeel.db"
    app_secret: str = "dev_secret"
    debug: bool = True


settings = Settings()
