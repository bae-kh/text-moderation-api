from fastapi import APIRouter, Request
from app.schemas.payload import DetectRequest, DetectResponse
from app.services.model import HateSpeechModel
from starlette.concurrency import run_in_threadpool
from fastapi import Depends
from sqlalchemy.orm import Session
import logging
import time

from app.core.logging import log_event
from app.db.database import get_db
from app.services.moderation_store import create_moderation_record

router = APIRouter()

@router.get("/health")
async def health_check() -> dict:
    return {
        "status": "ok",
        "message": "API server is running"
    }

@router.post("/detect", response_model=DetectResponse)
async def detect_text(
    request: DetectRequest,
    fastapi_req: Request,
    db: Session = Depends(get_db),
) -> DetectResponse:
    start_time = time.perf_counter()
    request_id = getattr(fastapi_req.state, "request_id", None)

    model: HateSpeechModel = fastapi_req.app.state.model

    try:
        result = await run_in_threadpool(model.predict, request.text)

        record = create_moderation_record(
            db=db,
            text=request.text,
            result=result,
        )

        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        log_event(
            event="detect_completed",
            request_id=request_id,
            method=fastapi_req.method,
            path=str(fastapi_req.url.path),
            status_code=200,
            latency_ms=latency_ms,
            text_length=len(request.text),
            category=result["category"],
            confidence=result["confidence"],
            action=result["action"],
            stored=record is not None,
        )

        return DetectResponse(
            is_hate_speech=result["is_hate_speech"],
            confidence=result["confidence"],
            category=result["category"],
            action=result["action"],
            message=result["message"],
        )

    except Exception as e:
        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        log_event(
            event="detect_failed",
            level=logging.ERROR,
            request_id=request_id,
            method=fastapi_req.method,
            path=str(fastapi_req.url.path),
            latency_ms=latency_ms,
            text_length=len(request.text),
            error_type=type(e).__name__,
            error_message=str(e),
        )

        raise
