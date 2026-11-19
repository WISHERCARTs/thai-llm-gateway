"""Settings read from the environment (or a .env file). Secrets come from Secret Manager on Cloud Run."""

from functools import cached_property

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    service_name: str = "thai-llm-gateway"
    environment: str = "local"

    # "name:key,name2:key2". Each client sends the key as a bearer token.
    gateway_client_keys: SecretStr = SecretStr("")

    # Key for hashing end-user ids (IPs) in logs and rate-limit keys.
    user_hash_salt: SecretStr = SecretStr("")

    gemini_api_key: SecretStr = SecretStr("")
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    gemini_models: list[str] = Field(default_factory=lambda: ["gemini-2.5-flash", "gemini-2.5-flash-lite"])
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_enabled: bool = True
    upstream_timeout_seconds: float = 60.0
    upstream_retries: int = 2

    rate_limit_client_per_minute: int = 120
    rate_limit_user_per_minute: int = 5
    rate_limit_user_per_day: int = 30
    daily_budget_usd: float = 1.0

    upstash_redis_rest_url: str = ""
    upstash_redis_rest_token: SecretStr = SecretStr("")
    # Postgres for the usage log (Neon in production). Empty: keep records in memory.
    database_url: SecretStr = SecretStr("")

    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_public_key: str = ""
    langfuse_secret_key: SecretStr = SecretStr("")

    pii_redaction: bool = True
    # Put the caller's own values back into the answer after the model replied with placeholders.
    pii_restore: bool = True
    injection_mode: str = "block"  # block | flag | off
    injection_threshold: float = 0.7
    system_leak_check: bool = True

    @cached_property
    def client_keys(self) -> dict[str, str]:
        """Map of key -> client name."""
        out: dict[str, str] = {}
        for pair in self.gateway_client_keys.get_secret_value().split(","):
            name, sep, key = pair.strip().partition(":")
            if sep and name and key:
                out[key] = name
        return out
