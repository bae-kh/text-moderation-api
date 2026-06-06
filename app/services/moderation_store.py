from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import ModerationRecord


RECORDABLE_ACTIONS = {"block", "review"}


def create_moderation_record(
    db: Session,
    text: str,
    result: dict,
) -> ModerationRecord | None:
    action = result["action"]

    if action not in RECORDABLE_ACTIONS:
        return None

    record = ModerationRecord(
        text=text,
        is_hate_speech=result["is_hate_speech"],
        category=result["category"],
        confidence=result["confidence"],
        action=action,
        status="pending",
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    return record


def list_moderation_records(
    db: Session,
    status: str | None = None,
    action: str | None = None,
    category: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[ModerationRecord], int]:
    """
    pagination과 filter를 지원하는 moderation records 조회.

    Returns:
        (items, total) 튜플
    """
    # 기본 필터 조건 구성
    filters = []

    if status:
        filters.append(ModerationRecord.status == status)

    if action:
        filters.append(ModerationRecord.action == action)

    if category:
        filters.append(ModerationRecord.category == category)

    # count 쿼리
    count_stmt = select(func.count()).select_from(ModerationRecord)
    for f in filters:
        count_stmt = count_stmt.where(f)
    total = db.scalar(count_stmt) or 0

    # items 쿼리
    items_stmt = select(ModerationRecord)
    for f in filters:
        items_stmt = items_stmt.where(f)
    items_stmt = (
        items_stmt
        .order_by(ModerationRecord.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    items = list(db.scalars(items_stmt).all())

    return items, total


def get_moderation_record(
    db: Session,
    record_id: int,
) -> ModerationRecord | None:
    return db.get(ModerationRecord, record_id)


def update_moderation_review(
    db: Session,
    record: ModerationRecord,
    review_result: str,
    review_note: str | None,
) -> ModerationRecord:
    record.status = "resolved"
    record.review_result = review_result
    record.review_note = review_note

    db.add(record)
    db.commit()
    db.refresh(record)

    return record
