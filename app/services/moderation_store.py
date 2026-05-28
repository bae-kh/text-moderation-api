from sqlalchemy import select
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
    limit: int = 20,
) -> list[ModerationRecord]:
    statement = select(ModerationRecord).order_by(ModerationRecord.created_at.desc())

    if status:
        statement = statement.where(ModerationRecord.status == status)

    if action:
        statement = statement.where(ModerationRecord.action == action)

    statement = statement.limit(limit)

    return list(db.scalars(statement).all())


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
