import logging
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import ModelInferenceError

logger = logging.getLogger(__name__)

settings = get_settings()


class HateSpeechModel:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.model_name
        self.pipeline = None

    @property
    def is_loaded(self) -> bool:
        """모델이 메모리에 적재되어 있는지 확인합니다."""
        return self.pipeline is not None

    def load(self) -> None:
        """HuggingFace 모델을 메모리에 적재합니다."""
        from transformers import pipeline as hf_pipeline

        logger.info(f"Loading model {self.model_name} into memory...")
        try:
            self.pipeline = hf_pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,
                function_to_apply="softmax",
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")

    def predict(self, text: str) -> dict[str, Any]:
        """적재된 모델을 통해 신고 텍스트/댓글의 유해 표현 여부를 추론합니다."""
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")

        try:
            results = self.pipeline(
                text,
                truncation=True,
                max_length=settings.model_max_tokens,
            )

            if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
                results = results[0]

            best_result = max(results, key=lambda item: item["score"])

            label = best_result["label"]
            score = float(best_result["score"])

            is_hate_speech = label != "clean"
            action = self._decide_action(label=label, confidence=score)
            message = self._build_message(action=action)

            return {
                "is_hate_speech": is_hate_speech,
                "confidence": score,
                "category": label,
                "action": action,
                "message": message,
            }

        except RuntimeError as e:
            logger.error(f"Runtime inference error: {e}")
            raise ModelInferenceError(
                f"Internal error during model inference: {e}"
            )
        except Exception as e:
            logger.error(f"Unexpected inference error: {e}")
            raise ModelInferenceError(
                f"Unexpected error during model inference: {e}"
            )

    def _decide_action(self, label: str, confidence: float) -> str:
        if label == "clean":
            if confidence >= settings.clean_allow_threshold:
                return "allow"
            return "review"

        if confidence >= settings.harmful_block_threshold:
            return "block"

        return "review"

    def _build_message(self, action: str) -> str:
        if action == "allow":
            return "Message allowed."

        if action == "block":
            return "Message blocked due to harmful content."

        return "Message requires human review."

    def unload(self) -> None:
        logger.info("Unloading model and freeing resources...")
        self.pipeline = None
