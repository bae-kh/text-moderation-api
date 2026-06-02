from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


class FakeHateSpeechModel:
    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    @property
    def is_loaded(self) -> bool:
        return True

    def predict(self, text: str) -> dict[str, Any]:
        return {
            "is_hate_speech": False,
            "confidence": 0.99,
            "category": "clean",
            "action": "allow",
            "message": "Message allowed.",
        }


@patch("app.main.HateSpeechModel", return_value=FakeHateSpeechModel())
def test_health_check(mock_model: Any) -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "db_connected" in data
    assert "x-request-id" in response.headers
    assert len(response.headers["x-request-id"]) > 0


@patch("app.main.HateSpeechModel", return_value=FakeHateSpeechModel())
def test_detect_success(mock_model: Any) -> None:
    payload: dict[str, str] = {"text": "신고된 댓글 예시입니다."}

    with TestClient(app) as client:
        response = client.post("/api/v1/detect", json=payload)

    assert response.status_code == 200
    assert "x-request-id" in response.headers
    assert len(response.headers["x-request-id"]) > 0

    data: dict[str, Any] = response.json()

    assert data["is_hate_speech"] is False
    assert data["confidence"] == 0.99
    assert data["category"] == "clean"
    assert data["action"] == "allow"
    assert data["message"] == "Message allowed."


@patch("app.main.HateSpeechModel", return_value=FakeHateSpeechModel())
def test_detect_empty_text_validation(mock_model: Any) -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/detect", json={"text": ""})

    assert response.status_code == 422
    assert "x-request-id" in response.headers


@patch("app.main.HateSpeechModel", return_value=FakeHateSpeechModel())
def test_detect_blank_text_validation(mock_model: Any) -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/detect", json={"text": "   "})

    assert response.status_code == 422
    assert "x-request-id" in response.headers


@patch("app.main.HateSpeechModel", return_value=FakeHateSpeechModel())
def test_detect_missing_text_validation(mock_model: Any) -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/detect", json={})

    assert response.status_code == 422
    assert "x-request-id" in response.headers


@patch("app.main.HateSpeechModel", return_value=FakeHateSpeechModel())
def test_detect_too_long_text_validation(mock_model: Any) -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/detect", json={"text": "a" * 1001})

    assert response.status_code == 422
    assert "x-request-id" in response.headers