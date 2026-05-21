from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class LLMProvider(str, Enum):
    """Доступные LLM провайдеры"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ChatRequest(BaseModel):
    """Упрощенный запрос для чата"""
    message: str
    document_ids: Optional[List[str]] = None  # Конкретные документы для поиска
    max_relevant_chunks: Optional[int] = None
    similarity_threshold: Optional[float] = None
    temperature: Optional[float] = None

    # Новые параметры для выбора модели
    llm_provider: Optional[LLMProvider] = None  # openai или anthropic
    # конкретная модель (gpt-4, claude-3-sonnet-20240229, etc.)
    model_name: Optional[str] = None


class ChatResponse(BaseModel):
    """Упрощенный ответ чата"""
    message_id: str
    session_id: str  # Заглушка для совместимости
    content: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RelevantChunk(BaseModel):
    """Релевантный чанк документа"""
    chunk_id: str
    document_id: str
    document_title: str
    content: str  # Полный контент без обрезания
    similarity: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
