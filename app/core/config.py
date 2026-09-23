"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings object for AgentOS."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AgentOS"
    app_env: str = "development"
    debug: bool = True

    # --- LLM Provider selection ---
    llm_provider: str = "ollama"          # runtime
    eval_llm_provider: str = "ollama"     # used by eval runs

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    # --- Groq ---
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"

    # --- Gemini ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # --- Database ---
    database_url: str = "sqlite:///./agentos.db"


settings = Settings()