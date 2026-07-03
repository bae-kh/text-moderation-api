from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import get_settings


settings = get_settings()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_admin_api_key(
    x_api_key: str | None = Security(api_key_header),
) -> None:
    """
    관리자 API 인증을 위한 FastAPI dependency입니다.

    요청 헤더의 X-API-Key 값이 서버에 설정된 ADMIN_API_KEY와 일치하는지 확인합니다.
    - 키가 없으면 401 Unauthorized
    - 키가 틀리면 403 Forbidden
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key.",
        )

    if x_api_key != settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )
