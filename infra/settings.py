from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    """
    pydantic BaseSettings automatically reads from .env file.
    each field name must match the key in .env exactly.
    if a key is missing from .emv it uses the default value.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # ── Postgres ──────────────────────────────────────────────
    neon_database_url: str = ""

    # ── Redis ─────────────────────────────────────────────────
    upstash_redis_url: str = ""

    # ── Kafka ─────────────────────────────────────────────────
    # local docker kafka - no auth needed
    kafka_bootstrap_servers: str = "localhost:9092"

    # ── Qdrant ────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    # ── LLMs ──────────────────────────────────────────────────
    groq_api_key: str = ""
    openai_api_key: str = ""

    # ── Voice ─────────────────────────────────────────────────
    deepgram_api_key: str = ""
    elevenlabs_api_key: str = ""

    # ── GCS ───────────────────────────────────────────────────
    gcs_bucket_name: str = "capitalsense-filings"
    google_application_credentials: str = "./gcp-key.json"

    # ── Observability ─────────────────────────────────────────
    otel_endpoint: str = "http://localhost:4317"
    prometheus_port: int = 8001

    # ── App ───────────────────────────────────────────────────
    environment: str = "development"
    log_level: str = "INFO"

    @property 
    def is_dev(self) -> bool:
        """return true when running locally, False in production"""
        return self.environment == "development"

@lru_cache
def get_settings() -> Settings:
    return Settings()