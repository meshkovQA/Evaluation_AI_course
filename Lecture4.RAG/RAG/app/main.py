from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import uvicorn
from datetime import datetime
from contextlib import asynccontextmanager

from app.config import settings, get_storage_paths, validate_settings
from app.api.documents import router as documents_router
from app.api.chat import router as chat_router


# Настройка логирования
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    # Startup
    logger.info("Запуск RAG System API...")

    try:
        # Проверяем настройки
        validate_settings()
        logger.info("✓ Настройки валидированы")

        # Создаем необходимые директории
        get_storage_paths()
        logger.info("✓ Директории для хранения созданы")

        logger.info("🚀 RAG System API успешно запущен")

    except Exception as e:
        logger.error(f"❌ Ошибка при запуске: {e}")
        raise

    yield

    # Shutdown
    logger.info("Завершение работы RAG System API...")


# Создаем приложение FastAPI
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="RAG System API для загрузки документов и чата",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Настройка CORS - РАЗРЕШАЕМ ВСЕ
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешаем все источники
    allow_credentials=True,
    allow_methods=["*"],  # Разрешаем все методы
    allow_headers=["*"],  # Разрешаем все заголовки
)


# Основные роуты
@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "message": "RAG System API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Проверка состояния системы"""
    try:
        services_status = {}

        # Проверяем embedding сервис
        try:
            from app.services.embedding_service import get_embedding_service
            embedding_service = get_embedding_service()
            services_status["embeddings"] = await embedding_service.test_connection()
        except Exception as e:
            services_status["embeddings"] = False
            logger.error(f"Embedding service error: {e}")

        # Проверяем векторное хранилище
        try:
            from app.services.vector_store import get_vector_store_service
            vector_store = get_vector_store_service()
            services_status["vector_store"] = await vector_store.test_connection()
        except Exception as e:
            services_status["vector_store"] = False
            logger.error(f"Vector store error: {e}")

        # Проверяем LLM сервис
        try:
            from app.services.llm_service import get_llm_service
            llm_service = get_llm_service()
            services_status["llm"] = await llm_service.test_connection()
        except Exception as e:
            services_status["llm"] = False
            logger.error(f"LLM service error: {e}")

        # Определяем общий статус
        all_healthy = all(services_status.values())
        status_code = 200 if all_healthy else 503

        return JSONResponse(
            status_code=status_code,
            content={
                "status": "healthy" if all_healthy else "degraded",
                "version": settings.APP_VERSION,
                "timestamp": datetime.utcnow().isoformat(),
                "services": {
                    "api": "running",
                    "storage": "available",
                    "embeddings": "connected" if services_status.get("embeddings") else "disconnected",
                    "vector_store": "connected" if services_status.get("vector_store") else "disconnected",
                    "llm": "connected" if services_status.get("llm") else "disconnected"
                }
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@app.get("/info")
async def get_info():
    """Информация о системе"""
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "supported_formats": settings.ALLOWED_FILE_EXTENSIONS,
        "max_file_size": f"{settings.MAX_FILE_SIZE / (1024 * 1024):.1f} MB",
        "chunk_size": settings.DEFAULT_CHUNK_SIZE,
        "chunk_overlap": settings.DEFAULT_CHUNK_OVERLAP,
        "text_splitter": settings.TEXT_SPLITTER_TYPE,
        "llm_provider": settings.DEFAULT_LLM_PROVIDER,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "vector_db": settings.VECTOR_DB_TYPE
    }


# Подключаем роуты
app.include_router(documents_router, prefix=settings.API_PREFIX)
app.include_router(chat_router, prefix=settings.API_PREFIX)


if __name__ == "__main__":
    # Запуск для разработки
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )