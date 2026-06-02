class ModelInferenceError(Exception):
    """
    모델 추론 중 발생하는 비즈니스 예외입니다.

    서비스 레이어에서 FastAPI의 HTTPException 대신 이 예외를 사용하여,
    서비스 레이어가 HTTP 응답 형식에 직접 의존하지 않도록 합니다.

    라우터 계층에서 이 예외를 캐치하여 적절한 HTTPException으로 변환합니다.
    """

    def __init__(self, message: str = "Model inference failed."):
        self.message = message
        super().__init__(self.message)
