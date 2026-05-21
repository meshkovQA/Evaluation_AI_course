from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
import uuid


class DocumentStatus(str, Enum):
    """Статусы обработки документа"""
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class DocumentType(str, Enum):
    """Типы документов"""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    CSV = "csv"
    JSON = "json"
    HTML = "html"
    MARKDOWN = "md"
    EXCEL = "xlsx"


class DocumentMetadata(BaseModel):
    """Метаданные документа"""
    filename: str
    file_size: int
    file_extension: str
    created_at: float
    modified_at: float
    mime_type: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    char_count: Optional[int] = None


class DocumentChunk(BaseModel):
    """Фрагмент документа для векторного поиска"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    chunk_index: int
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

    class Config:
        arbitrary_types_allowed = True


class Document(BaseModel):
    """Основная модель документа"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str = ""
    file_path: str
    document_type: DocumentType
    status: DocumentStatus = DocumentStatus.UPLOADING
    metadata: DocumentMetadata
    chunks: List[DocumentChunk] = Field(default_factory=list)

    # Временные метки
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None

    # Дополнительные поля
    tags: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    error_message: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class DocumentUploadResponse(BaseModel):
    """Ответ при загрузке документа"""
    document_id: str
    filename: str
    status: DocumentStatus
    message: str


class DocumentListResponse(BaseModel):
    """Ответ со списком документов"""
    documents: List[Document]
    total: int
    page: int
    page_size: int


class DocumentDetailResponse(BaseModel):
    """Детальная информация о документе"""
    document: Document
    chunks_count: int
    processing_info: Optional[Dict[str, Any]] = None


# Запросы для API
class DocumentUploadRequest(BaseModel):
    """Запрос на загрузку документа"""
    title: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class DocumentUpdateRequest(BaseModel):
    """Запрос на обновление документа"""
    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None


class DocumentSearchRequest(BaseModel):
    """Запрос для поиска документов"""
    query: Optional[str] = None
    document_type: Optional[DocumentType] = None
    tags: Optional[List[str]] = None
    status: Optional[DocumentStatus] = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
