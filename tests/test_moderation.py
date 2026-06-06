from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import Session, sessionmaker

from app.main import app
from app.db.database import Base, get_db


ADMIN_HEADERS = {"X-API-Key": "dev-admin-key"}


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


# ─── 기존 기능 테스트 (Auth 헤더 추가) ─────────────────────


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

    records_response = client.get(
        "/api/v1/moderation/records",
        headers=ADMIN_HEADERS,
    )

    assert records_response.status_code == 200
    assert records_response.json()["items"] == []
    assert records_response.json()["total"] == 0


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

    records_response = client.get(
        "/api/v1/moderation/records",
        headers=ADMIN_HEADERS,
    )

    assert records_response.status_code == 200

    data = records_response.json()

    assert data["total"] == 1
    assert len(data["items"]) == 1
    assert data["items"][0]["text"] == "바보같아"
    assert data["items"][0]["category"] == "악플/욕설"
    assert data["items"][0]["action"] == "block"
    assert data["items"][0]["status"] == "pending"
    assert data["items"][0]["review_result"] is None
    assert data["items"][0]["review_note"] is None


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

    records_response = client.get(
        "/api/v1/moderation/records",
        headers=ADMIN_HEADERS,
    )
    record_id = records_response.json()["items"][0]["id"]

    detail_response = client.get(
        f"/api/v1/moderation/records/{record_id}",
        headers=ADMIN_HEADERS,
    )

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

    records_response = client.get(
        "/api/v1/moderation/records",
        headers=ADMIN_HEADERS,
    )
    record_id = records_response.json()["items"][0]["id"]

    patch_response = client.patch(
        f"/api/v1/moderation/records/{record_id}",
        json={
            "review_result": "confirmed_harmful",
            "review_note": "운영자 검토 결과 유해 표현으로 판단됨",
        },
        headers=ADMIN_HEADERS,
    )

    assert patch_response.status_code == 200

    patch_data = patch_response.json()

    assert patch_data["id"] == record_id
    assert patch_data["status"] == "resolved"
    assert patch_data["review_result"] == "confirmed_harmful"
    assert patch_data["review_note"] == "운영자 검토 결과 유해 표현으로 판단됨"

    detail_response = client.get(
        f"/api/v1/moderation/records/{record_id}",
        headers=ADMIN_HEADERS,
    )

    assert detail_response.status_code == 200

    detail_data = detail_response.json()

    assert detail_data["status"] == "resolved"
    assert detail_data["review_result"] == "confirmed_harmful"
    assert detail_data["review_note"] == "운영자 검토 결과 유해 표현으로 판단됨"


def test_get_moderation_record_not_found_returns_404(client: TestClient) -> None:
    response = client.get(
        "/api/v1/moderation/records/9999",
        headers=ADMIN_HEADERS,
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Moderation record not found."


# ─── Auth 테스트 ─────────────────────────────────────────


def test_list_records_without_api_key_returns_401(client: TestClient) -> None:
    response = client.get("/api/v1/moderation/records")

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing API key."


def test_list_records_with_wrong_api_key_returns_403(client: TestClient) -> None:
    response = client.get(
        "/api/v1/moderation/records",
        headers={"X-API-Key": "wrong-key"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid API key."


def test_get_record_without_api_key_returns_401(client: TestClient) -> None:
    response = client.get("/api/v1/moderation/records/1")

    assert response.status_code == 401


def test_patch_record_without_api_key_returns_401(client: TestClient) -> None:
    response = client.patch(
        "/api/v1/moderation/records/1",
        json={"review_result": "confirmed_harmful"},
    )

    assert response.status_code == 401


# ─── Pagination 테스트 ───────────────────────────────────


def _create_block_records(client: TestClient, count: int) -> None:
    """테스트용 block record를 count개 생성합니다."""
    app.state.model = FakeModerationModel(
        result={
            "is_hate_speech": True,
            "confidence": 0.91,
            "category": "hate",
            "action": "block",
            "message": "Message blocked due to harmful content.",
        }
    )

    for i in range(count):
        client.post(
            "/api/v1/detect",
            json={"text": f"유해 댓글 {i}"},
        )


def test_pagination_limit_and_offset(client: TestClient) -> None:
    _create_block_records(client, 5)

    response = client.get(
        "/api/v1/moderation/records?limit=2&offset=0",
        headers=ADMIN_HEADERS,
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["items"]) == 2
    assert data["total"] == 5
    assert data["limit"] == 2
    assert data["offset"] == 0


def test_pagination_offset_skips_records(client: TestClient) -> None:
    _create_block_records(client, 5)

    response = client.get(
        "/api/v1/moderation/records?limit=2&offset=4",
        headers=ADMIN_HEADERS,
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["items"]) == 1
    assert data["total"] == 5


def test_pagination_default_values(client: TestClient) -> None:
    _create_block_records(client, 3)

    response = client.get(
        "/api/v1/moderation/records",
        headers=ADMIN_HEADERS,
    )

    data = response.json()

    assert data["limit"] == 20
    assert data["offset"] == 0
    assert data["total"] == 3
    assert len(data["items"]) == 3


# ─── Filter 테스트 ───────────────────────────────────────


def test_filter_by_status(client: TestClient) -> None:
    app.state.model = FakeModerationModel(
        result={
            "is_hate_speech": True,
            "confidence": 0.91,
            "category": "hate",
            "action": "block",
            "message": "Message blocked due to harmful content.",
        }
    )

    # 2개 생성 (둘 다 pending)
    client.post("/api/v1/detect", json={"text": "댓글 1"})
    client.post("/api/v1/detect", json={"text": "댓글 2"})

    # 1개를 resolved로 변경
    records = client.get(
        "/api/v1/moderation/records",
        headers=ADMIN_HEADERS,
    ).json()["items"]

    client.patch(
        f"/api/v1/moderation/records/{records[0]['id']}",
        json={"review_result": "confirmed_harmful"},
        headers=ADMIN_HEADERS,
    )

    # pending만 조회
    pending_response = client.get(
        "/api/v1/moderation/records?status=pending",
        headers=ADMIN_HEADERS,
    )

    assert pending_response.json()["total"] == 1
    assert pending_response.json()["items"][0]["status"] == "pending"

    # resolved만 조회
    resolved_response = client.get(
        "/api/v1/moderation/records?status=resolved",
        headers=ADMIN_HEADERS,
    )

    assert resolved_response.json()["total"] == 1
    assert resolved_response.json()["items"][0]["status"] == "resolved"


def test_filter_by_action(client: TestClient) -> None:
    # block record 생성
    app.state.model = FakeModerationModel(
        result={
            "is_hate_speech": True,
            "confidence": 0.91,
            "category": "hate",
            "action": "block",
            "message": "Blocked.",
        }
    )
    client.post("/api/v1/detect", json={"text": "블록 댓글"})

    # review record 생성
    app.state.model = FakeModerationModel(
        result={
            "is_hate_speech": True,
            "confidence": 0.55,
            "category": "hate",
            "action": "review",
            "message": "Review.",
        }
    )
    client.post("/api/v1/detect", json={"text": "리뷰 댓글"})

    # block만 필터
    block_response = client.get(
        "/api/v1/moderation/records?action=block",
        headers=ADMIN_HEADERS,
    )

    assert block_response.json()["total"] == 1
    assert block_response.json()["items"][0]["action"] == "block"


def test_filter_by_category(client: TestClient) -> None:
    # hate category
    app.state.model = FakeModerationModel(
        result={
            "is_hate_speech": True,
            "confidence": 0.91,
            "category": "hate",
            "action": "block",
            "message": "Blocked.",
        }
    )
    client.post("/api/v1/detect", json={"text": "혐오 댓글"})

    # insult category
    app.state.model = FakeModerationModel(
        result={
            "is_hate_speech": True,
            "confidence": 0.88,
            "category": "insult",
            "action": "block",
            "message": "Blocked.",
        }
    )
    client.post("/api/v1/detect", json={"text": "모욕 댓글"})

    # hate만 필터
    response = client.get(
        "/api/v1/moderation/records?category=hate",
        headers=ADMIN_HEADERS,
    )

    assert response.json()["total"] == 1
    assert response.json()["items"][0]["category"] == "hate"
