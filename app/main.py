import uuid
import logging
from contextlib import asynccontextmanager
from typing import Callable, Awaitable, AsyncGenerator
from fastapi import FastAPI, Request, Response
from app.api.routes import router as api_router
from app.services.model import HateSpeechModel
from app.core.config import get_settings
from app.db.database import init_db
from app.api.moderation import router as moderation_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    서버의 시작과 종료 생명주기를 관리합니다. (Lifespan Context Manager)
    """
    settings = get_settings()

    # SQLite(로컬/Docker 단독)에서는 create_all로 테이블 생성
    # PostgreSQL(Docker Compose)에서는 alembic upgrade head가 schema를 관리
    if settings.database_url.startswith("sqlite"):
        logger.info("SQLite detected. Creating tables via create_all.")
        init_db()
    else:
        logger.info("Non-SQLite DB detected. Skipping create_all (managed by Alembic).")
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
    요청 단위 추적을 위해 request_id를 생성하고 응답 헤더와 로그 컨텍스트에 전달합니다.
    """
    request_id: str = str(uuid.uuid4())
    request.state.request_id = request_id

    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = request_id

    return response

# 라우터 등록
app.include_router(api_router, prefix="/api/v1")
app.include_router(moderation_router, prefix="/api/v1")
