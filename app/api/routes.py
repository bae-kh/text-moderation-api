from fastapi import APIRouter, Request
from app.schemas.payload import DetectRequest, DetectResponse
from app.services.model import HateSpeechModel
from starlette.concurrency import run_in_threadpool
from fastapi import Depends
from sqlalchemy.orm import Session

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
    model: HateSpeechModel = fastapi_req.app.state.model

    result = await run_in_threadpool(model.predict, request.text)

    create_moderation_record(
        db=db,
        text=request.text,
        result=result,
    )

    return DetectResponse(
        is_hate_speech=result["is_hate_speech"],
        confidence=result["confidence"],
        category=result["category"],
        action=result["action"],
        message=result["message"],
    )
