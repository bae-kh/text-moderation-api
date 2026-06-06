from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModerationRecordResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    text: str
    is_hate_speech: bool
    category: str
    confidence: float
    action: str
    status: str
    review_result: str | None
    review_note: str | None
    created_at: datetime
    updated_at: datetime


class ModerationReviewUpdateRequest(BaseModel):
    review_result: Literal["confirmed_harmful", "false_positive", "clean"] = Field(
        ...,
        description="운영자의 최종 검토 결과",
    )
    review_note: str | None = Field(
        default=None,
        max_length=1000,
        description="운영자 검토 메모",
    )


class ModerationReviewUpdateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    status: str
    review_result: str
    review_note: str | None


class ModerationRecordListResponse(BaseModel):
    items: list[ModerationRecordResponse]
    total: int
    limit: int
    offset: int
