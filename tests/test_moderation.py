from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import Session, sessionmaker

from app.main import app
from app.db.database import Base, get_db


class FakeModerationModel:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    @property
    def is_loaded(self) -> bool:
        return True

    def predict(self, text: str) -> dict[str, Any]:
        return self.result


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )

    Base.metadata.create_all(bind=engine)

    def override_get_db() -> Generator[Session, None, None]:
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)


def test_detect_allow_does_not_create_moderation_record(client: TestClient) -> None:
    app.state.model = FakeModerationModel(
        result={
            "is_hate_speech": False,
            "confidence": 0.95,
            "category": "clean",
            "action": "allow",
            "message": "Message allowed.",
        }
    )

    response = client.post(
        "/api/v1/detect",
        json={"text": "정상 댓글입니다."},
    )

    assert response.status_code == 200
    assert response.json()["action"] == "allow"

    records_response = client.get("/api/v1/moderation/records")

    assert records_response.status_code == 200
    assert records_response.json() == []


def test_detect_block_creates_pending_moderation_record(client: TestClient) -> None:
    app.state.model = FakeModerationModel(
        result={
            "is_hate_speech": True,
            "confidence": 0.91,
            "category": "악플/욕설",
            "action": "block",
            "message": "Message blocked due to harmful content.",
        }
    )

    response = client.post(
        "/api/v1/detect",
        json={"text": "바보같아"},
    )

    assert response.status_code == 200
    assert response.json()["action"] == "block"

    records_response = client.get("/api/v1/moderation/records")

    assert records_response.status_code == 200

    records = records_response.json()

    assert len(records) == 1
    assert records[0]["text"] == "바보같아"
    assert records[0]["category"] == "악플/욕설"
    assert records[0]["action"] == "block"
    assert records[0]["status"] == "pending"
    assert records[0]["review_result"] is None
    assert records[0]["review_note"] is None


def test_get_moderation_record_detail(client: TestClient) -> None:
    app.state.model = FakeModerationModel(
        result={
            "is_hate_speech": True,
            "confidence": 0.91,
            "category": "악플/욕설",
            "action": "block",
            "message": "Message blocked due to harmful content.",
        }
    )

    detect_response = client.post(
        "/api/v1/detect",
        json={"text": "검토 대상 댓글입니다."},
    )
    assert detect_response.status_code == 200

    records_response = client.get("/api/v1/moderation/records")
    record_id = records_response.json()[0]["id"]

    detail_response = client.get(f"/api/v1/moderation/records/{record_id}")

    assert detail_response.status_code == 200

    data = detail_response.json()

    assert data["id"] == record_id
    assert data["text"] == "검토 대상 댓글입니다."
    assert data["action"] == "block"
    assert data["status"] == "pending"


def test_patch_moderation_record_resolves_review(client: TestClient) -> None:
    app.state.model = FakeModerationModel(
        result={
            "is_hate_speech": True,
            "confidence": 0.91,
            "category": "악플/욕설",
            "action": "block",
            "message": "Message blocked due to harmful content.",
        }
    )

    detect_response = client.post(
        "/api/v1/detect",
        json={"text": "운영자 검토 대상입니다."},
    )
    assert detect_response.status_code == 200

    records_response = client.get("/api/v1/moderation/records")
    record_id = records_response.json()[0]["id"]

    patch_response = client.patch(
        f"/api/v1/moderation/records/{record_id}",
        json={
            "review_result": "confirmed_harmful",
            "review_note": "운영자 검토 결과 유해 표현으로 판단됨",
        },
    )

    assert patch_response.status_code == 200

    patch_data = patch_response.json()

    assert patch_data["id"] == record_id
    assert patch_data["status"] == "resolved"
    assert patch_data["review_result"] == "confirmed_harmful"
    assert patch_data["review_note"] == "운영자 검토 결과 유해 표현으로 판단됨"

    detail_response = client.get(f"/api/v1/moderation/records/{record_id}")

    assert detail_response.status_code == 200

    detail_data = detail_response.json()

    assert detail_data["status"] == "resolved"
    assert detail_data["review_result"] == "confirmed_harmful"
    assert detail_data["review_note"] == "운영자 검토 결과 유해 표현으로 판단됨"


def test_get_moderation_record_not_found_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/moderation/records/9999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Moderation record not found."
