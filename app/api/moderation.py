from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.security import verify_admin_api_key
from app.db.database import get_db
from app.schemas.moderation import (
    ModerationRecordListResponse,
    ModerationRecordResponse,
    ModerationReviewUpdateRequest,
    ModerationReviewUpdateResponse,
)
from app.services.moderation_store import (
    get_moderation_record,
    list_moderation_records,
    update_moderation_review,
)

router = APIRouter()


@router.get(
    "/moderation/records",
    response_model=ModerationRecordListResponse,
)
def get_records(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: str | None = Query(default=None),
    action: str | None = Query(default=None),
    category: str | None = Query(default=None),
    db: Session = Depends(get_db),
    _: None = Depends(verify_admin_api_key),
) -> ModerationRecordListResponse:
    items, total = list_moderation_records(
        db=db,
        status=status,
        action=action,
        category=category,
        limit=limit,
        offset=offset,
    )

    return ModerationRecordListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/moderation/records/{record_id}",
    response_model=ModerationRecordResponse,
)
def get_record(
    record_id: int,
    db: Session = Depends(get_db),
    _: None = Depends(verify_admin_api_key),
) -> ModerationRecordResponse:
    record = get_moderation_record(db=db, record_id=record_id)

    if not record:
        raise HTTPException(
            status_code=404,
            detail="Moderation record not found.",
        )

    return record


@router.patch(
    "/moderation/records/{record_id}",
    response_model=ModerationReviewUpdateResponse,
)
def update_record_review(
    record_id: int,
    payload: ModerationReviewUpdateRequest,
    db: Session = Depends(get_db),
    _: None = Depends(verify_admin_api_key),
) -> ModerationReviewUpdateResponse:
    record = get_moderation_record(db=db, record_id=record_id)

    if not record:
        raise HTTPException(
            status_code=404,
            detail="Moderation record not found.",
        )

    updated_record = update_moderation_review(
        db=db,
        record=record,
        review_result=payload.review_result,
        review_note=payload.review_note,
    )

    return updated_record
