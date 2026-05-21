import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Настройки приложения"""

    # Основные настройки
    APP_NAME: str = "RAG System API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # API настройки
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8002
    API_PREFIX: str = "/api/v1"

    # Хранилище документов
    DOCUMENTS_STORAGE_PATH: str = "storage/documents"
    VECTOR_DB_PATH: str = "storage/vector_db"

    # Ограничения загрузки файлов
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_FILE_EXTENSIONS: list = [
        ".pdf", ".docx", ".txt", ".csv",
        ".json", ".html", ".md", ".xlsx"
    ]

    # Настройки разделения текста
    DEFAULT_CHUNK_SIZE: int = 1000
    DEFAULT_CHUNK_OVERLAP: int = 200
    # recursive, sentence, paragraph, fixed, markdown
    TEXT_SPLITTER_TYPE: str = "recursive"

    # LLM настройки
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    DEFAULT_LLM_PROVIDER: str = "openai"  # openai, anthropic, local
    DEFAULT_LLM_MODEL: str = "gpt-4o-mini"

    # Embedding настройки
    EMBEDDING_PROVIDER: str = "openai"  # openai, huggingface, local
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    EMBEDDING_DIMENSION: int = 1536

    # Векторная база данных
    VECTOR_DB_TYPE: str = "chroma"  # chroma, pinecone, faiss
    CHROMA_PERSIST_DIRECTORY: str = "storage/vector_db/chroma"

    # Pinecone настройки (если используется)
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = None
    PINECONE_INDEX_NAME: str = "rag-documents"

    # CORS настройки
    CORS_ORIGINS: list = ["*"]
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: list = ["*"]
    CORS_HEADERS: list = ["*"]

    # Логирование
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # База данных (если будет использоваться)
    DATABASE_URL: Optional[str] = None

    # Redis (для кэширования, если нужно)
    REDIS_URL: Optional[str] = None

    # Безопасность
    SECRET_KEY: str = "supersecretkey"

    # RAG настройки
    MAX_RELEVANT_CHUNKS: int = 5
    SIMILARITY_THRESHOLD: float = 0.3
    MAX_CONTEXT_LENGTH: int = 4000

    # Асинхронная обработка
    MAX_CONCURRENT_UPLOADS: int = 5
    PROCESSING_TIMEOUT: int = 300  # 5 минут

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Создаем глобальный экземпляр настроек
settings = Settings()


def get_storage_paths():
    """Создает необходимые директории для хранения"""
    paths = [
        settings.DOCUMENTS_STORAGE_PATH,
        settings.VECTOR_DB_PATH,
        settings.CHROMA_PERSIST_DIRECTORY
    ]

    for path in paths:
        os.makedirs(path, exist_ok=True)

    return paths


def validate_settings():
    """Проверяет корректность настроек"""
    errors = []

    # Проверяем API ключи для LLM
    if settings.DEFAULT_LLM_PROVIDER == "openai" and not settings.OPENAI_API_KEY:
        errors.append(
            "OPENAI_API_KEY не установлен, но выбран провайдер OpenAI")

    if settings.DEFAULT_LLM_PROVIDER == "anthropic" and not settings.ANTHROPIC_API_KEY:
        errors.append(
            "ANTHROPIC_API_KEY не установлен, но выбран провайдер Anthropic")

    # Проверяем настройки embeddings
    if settings.EMBEDDING_PROVIDER == "openai" and not settings.OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY не установлен для embeddings")

    # Проверяем Pinecone настройки
    if settings.VECTOR_DB_TYPE == "pinecone":
        if not settings.PINECONE_API_KEY:
            errors.append("PINECONE_API_KEY не установлен")
        if not settings.PINECONE_ENVIRONMENT:
            errors.append("PINECONE_ENVIRONMENT не установлен")

    # Проверяем размеры чанков
    if settings.DEFAULT_CHUNK_SIZE <= settings.DEFAULT_CHUNK_OVERLAP:
        errors.append("CHUNK_SIZE должен быть больше CHUNK_OVERLAP")

    if errors:
        raise ValueError("Ошибки конфигурации:\n" + "\n".join(errors))

    return True


# Словарь для маппинга расширений файлов к MIME типам
FILE_TYPE_MAPPING = {
    '.pdf': 'application/pdf',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.doc': 'application/vnd.ms-word',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.xls': 'application/vnd.ms-excel',
    '.txt': 'text/plain',
    '.csv': 'text/csv',
    '.json': 'application/json',
    '.html': 'text/html',
    '.htm': 'text/html',
    '.md': 'text/markdown',
    '.markdown': 'text/x-markdown'
}


def get_file_mime_type(filename: str) -> Optional[str]:
    """Получает MIME тип файла по расширению"""
    ext = os.path.splitext(filename)[1].lower()
    return FILE_TYPE_MAPPING.get(ext)
