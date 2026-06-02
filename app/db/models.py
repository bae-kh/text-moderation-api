from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ModerationRecord(Base):
    __tablename__ = "moderation_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    text: Mapped[str] = mapped_column(Text, nullable=False)

    is_hate_speech: Mapped[bool] = mapped_column(nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    action: Mapped[str] = mapped_column(String(20), nullable=False)

    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    review_result: Mapped[str | None] = mapped_column(String(50), nullable=True)
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=_utc_now,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=_utc_now,
        onupdate=_utc_now,
        nullable=False,
    )
