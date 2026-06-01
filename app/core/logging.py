import json
import logging
from typing import Any


logger = logging.getLogger("app")


def log_event(
    event: str,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """
    Structured log를 JSON 문자열로 출력합니다.

    event:
        로그 이벤트 이름. 예: detect_completed, detect_failed

    fields:
        request_id, latency_ms, action 등 추가 필드
    """
    payload: dict[str, Any] = {
        "event": event,
        **fields,
    }

    logger.log(
        level,
        json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        ),
    )
