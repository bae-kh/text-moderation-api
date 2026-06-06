from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    애플리케이션 설정을 환경변수 또는 .env 파일에서 로딩합니다.

    하드코딩된 설정값을 외부화하여, 코드 변경 없이 운영 정책을 조정할 수 있도록 합니다.
    """

    # Database
    database_url: str = "sqlite:///./moderation.db"

    # Model
    model_name: str = "smilegate-ai/kor_unsmile"
    model_max_tokens: int = 256

    # Confidence Policy Thresholds
    clean_allow_threshold: float = 0.80
    harmful_block_threshold: float = 0.65

    # Logging
    log_level: str = "INFO"

    # Admin API key
    admin_api_key: str = "dev-admin-key"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        protected_namespaces=("settings_",),
    )


@lru_cache
def get_settings() -> Settings:
    """Settings 인스턴스를 캐싱하여 싱글톤으로 제공합니다."""
    return Settings()
