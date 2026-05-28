from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from app.services.model import HateSpeechModel


def test_model_predict_clean_high_confidence_allows() -> None:
    model = HateSpeechModel()
    model.pipeline = MagicMock(return_value=[
        {
            "label": "clean",
            "score": 0.91,
        }
    ])

    result: dict[str, Any] = model.predict("정상적인 댓글입니다.")

    assert result["is_hate_speech"] is False
    assert result["confidence"] == 0.91
    assert result["category"] == "clean"
    assert result["action"] == "allow"
    assert result["message"] == "Message allowed."


def test_model_predict_hate_high_confidence_blocks() -> None:
    model = HateSpeechModel()
    model.pipeline = MagicMock(return_value=[
        {
            "label": "hate",
            "score": 0.91,
        }
    ])

    result: dict[str, Any] = model.predict("유해 표현 예시")

    assert result["is_hate_speech"] is True
    assert result["confidence"] == 0.91
    assert result["category"] == "hate"
    assert result["action"] == "block"
    assert result["message"] == "Message blocked due to harmful content."


def test_model_predict_hate_low_confidence_requires_review() -> None:
    model = HateSpeechModel()
    model.pipeline = MagicMock(return_value=[
        {
            "label": "hate",
            "score": 0.64,
        }
    ])

    result: dict[str, Any] = model.predict("애매한 유해 표현 예시")

    assert result["is_hate_speech"] is True
    assert result["confidence"] == 0.64
    assert result["category"] == "hate"
    assert result["action"] == "review"
    assert result["message"] == "Message requires human review."


def test_model_predict_clean_low_confidence_requires_review() -> None:
    model = HateSpeechModel()
    model.pipeline = MagicMock(return_value=[
        {
            "label": "clean",
            "score": 0.72,
        }
    ])

    result: dict[str, Any] = model.predict("모델이 애매하게 clean으로 본 문장")

    assert result["is_hate_speech"] is False
    assert result["confidence"] == 0.72
    assert result["category"] == "clean"
    assert result["action"] == "review"
    assert result["message"] == "Message requires human review."


def test_model_predict_runtime_error_triggers_500() -> None:
    model = HateSpeechModel()
    model.pipeline = MagicMock(side_effect=RuntimeError("CPU inference error"))

    with pytest.raises(HTTPException) as exc_info:
        model.predict("긴 텍스트")

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Internal Server Error during model inference."

def test_model_predict_multi_label_output_blocks() -> None:
    model = HateSpeechModel()
    model.pipeline = MagicMock(return_value=[
        [
            {"label": "clean", "score": 0.03},
            {"label": "악플/욕설", "score": 0.91},
            {"label": "기타 혐오", "score": 0.02},
        ]
    ])

    result: dict[str, Any] = model.predict("공격적인 댓글 예시")

    assert result["is_hate_speech"] is True
    assert result["confidence"] == 0.91
    assert result["category"] == "악플/욕설"
    assert result["action"] == "block"
    assert result["message"] == "Message blocked due to harmful content."
