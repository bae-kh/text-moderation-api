import uuid
import logging
from contextlib import asynccontextmanager
from typing import Callable, Awaitable, AsyncGenerator
from fastapi import FastAPI, Request, Response
from app.api.routes import router as api_router
from app.services.model import HateSpeechModel
from app.db.database import init_db
from app.api.moderation import router as moderation_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    서버의 시작과 종료 생명주기를 관리합니다. (Lifespan Context Manager)
    """
    logger.info("Starting up server... Initializing database.")
    init_db()
    # Startup: 모델을 인스턴스화하고 메모리에 한 번만 적재(Singleton)
    logger.info("Starting up server... Loading AI Model.")
    model = HateSpeechModel()
    model.load()
    app.state.model = model

    yield

    # Shutdown: 자원 안전하게 해제
    logger.info("Shutting down server... Releasing AI Model resources.")
    model.unload()
    app.state.model = None

app = FastAPI(title="Text Filtering API", version="2.0.0-Phase2", lifespan=lifespan)

@app.middleware("http")
async def add_request_id_header(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """
    대용량 트래픽 환경에서의 추적성 확보를 위한 미들웨어입니다.
    """
    request_id: str = str(uuid.uuid4())
    request.state.request_id = request_id

    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = request_id

    return response

# 라우터 등록
app.include_router(api_router, prefix="/api/v1")
app.include_router(moderation_router, prefix="/api/v1")
