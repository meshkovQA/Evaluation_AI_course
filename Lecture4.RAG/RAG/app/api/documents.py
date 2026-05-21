from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from typing import List, Optional
from app.models.document import (
    Document, DocumentUploadRequest, DocumentUpdateRequest,
    DocumentUploadResponse, DocumentListResponse, DocumentDetailResponse,
    DocumentType, DocumentStatus, DocumentSearchRequest
)
from app.utils.auth import verify_api_key
from app.services.document_service import DocumentService, get_document_service as get_global_service

# Создаем роутер
router = APIRouter(prefix="/documents", tags=["documents"])

# Зависимость для получения сервиса документов
def get_document_service() -> DocumentService:
    return get_global_service()  # Используем глобальный экземпляр


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[str] = None,  # Теги через запятую
    service: DocumentService = Depends(get_document_service),
    api_key: str = Depends(verify_api_key)
):
    """
    Загружает новый документ

    - **file**: файл для загрузки (PDF, DOCX, TXT, CSV, JSON, HTML, MD, XLSX)
    - **title**: название документа (опционально)
    - **description**: описание документа (опционально)  
    - **tags**: теги через запятую (опционально)
    """

    if not api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # Обрабатываем теги
    tags_list = []
    if tags:
        tags_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

    request = DocumentUploadRequest(
        title=title,
        description=description,
        tags=tags_list
    )

    document = await service.upload_document(file, request)

    return DocumentUploadResponse(
        document_id=document.id,
        filename=document.metadata.filename,
        status=document.status,
        message="Документ успешно загружен и обрабатывается"
    )


@router.get("/", response_model=DocumentListResponse)
async def get_documents(
    page: int = Query(1, ge=1, description="Номер страницы"),
    page_size: int = Query(20, ge=1, le=100, description="Размер страницы"),
    status: Optional[DocumentStatus] = Query(
        None, description="Фильтр по статусу"),
    document_type: Optional[DocumentType] = Query(
        None, description="Фильтр по типу документа"),
    service: DocumentService = Depends(get_document_service),
    api_key: str = Depends(verify_api_key)
):
    """
    Получает список всех документов с пагинацией и фильтрацией

    - **page**: номер страницы (начиная с 1)
    - **page_size**: количество документов на странице
    - **status**: фильтр по статусу (uploading, processing, ready, error)
    - **document_type**: фильтр по типу документа
    """

    if not api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    result = service.get_documents(
        page=page,
        page_size=page_size,
        status=status,
        document_type=document_type
    )

    return DocumentListResponse(
        documents=result['documents'],
        total=result['total'],
        page=page,
        page_size=page_size
    )


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
    api_key: str = Depends(verify_api_key)
):
    """
    Получает детальную информацию о документе

    - **document_id**: ID документа
    """

    if not api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    document = service.get_document(document_id)

    if not document:
        raise HTTPException(status_code=404, detail="Документ не найден")

    return DocumentDetailResponse(
        document=document,
        chunks_count=len(document.chunks),
        processing_info={
            "total_chunks": len(document.chunks),
            "status": document.status.value,
            "processed_at": document.processed_at
        }
    )


@router.put("/{document_id}", response_model=Document)
async def update_document(
    document_id: str,
    request: DocumentUpdateRequest,
    service: DocumentService = Depends(get_document_service),
    api_key: str = Depends(verify_api_key)
):
    """
    Обновляет метаданные документа

    - **document_id**: ID документа
    - **title**: новое название
    - **description**: новое описание
    - **tags**: новые теги
    """
    if not api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    document = service.update_document(document_id, request)

    if not document:
        raise HTTPException(status_code=404, detail="Документ не найден")

    return document


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
    api_key: str = Depends(verify_api_key)

):
    """
    Удаляет документ и связанный файл

    - **document_id**: ID документа
    """
    if not api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    success = service.delete_document(document_id)

    if not success:
        raise HTTPException(status_code=404, detail="Документ не найден")

    return {"message": "Документ успешно удален"}


@router.get("/search/")
async def search_documents(
    query: str = Query(..., description="Поисковый запрос"),
    service: DocumentService = Depends(get_document_service),
    api_key: str = Depends(verify_api_key)
):
    """
    Поиск документов по содержимому и метаданным

    - **query**: поисковый запрос
    """
    if not api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    documents = service.search_documents(query)

    return {
        "query": query,
        "results": documents,
        "total": len(documents)
    }


@router.get("/{document_id}/chunks")
async def get_document_chunks(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
    api_key: str = Depends(verify_api_key)

):
    """
    Получает чанки документа для векторного поиска

    - **document_id**: ID документа
    """
    if not api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    

    document = service.get_document(document_id)

    if not document:
        raise HTTPException(status_code=404, detail="Документ не найден")

    chunks = await service.get_document_chunks(document_id)

    return {
        "document_id": document_id,
        "document_title": document.title,
        "chunks": chunks,
        "total_chunks": len(chunks)
    }


@router.get("/{document_id}/content")
async def get_document_content(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
    api_key: str = Depends(verify_api_key)
):
    """
    Получает полное содержимое документа

    - **document_id**: ID документа
    """
    if not api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    document = service.get_document(document_id)

    if not document:
        raise HTTPException(status_code=404, detail="Документ не найден")

    if document.status != DocumentStatus.READY:
        raise HTTPException(
            status_code=400,
            detail=f"Документ еще не обработан. Статус: {document.status.value}"
        )

    return {
        "document_id": document_id,
        "title": document.title,
        "content": document.content,
        "metadata": document.metadata,
        "word_count": document.metadata.word_count,
        "char_count": document.metadata.char_count
    }


@router.post("/bulk-upload")
async def bulk_upload_documents(
    files: List[UploadFile] = File(...),
    service: DocumentService = Depends(get_document_service),
    api_key: str = Depends(verify_api_key)

):
    """
    Массовая загрузка документов

    - **files**: список файлов для загрузки
    """
    if not api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    results = []

    for file in files:
        try:
            request = DocumentUploadRequest(title=None)
            document = await service.upload_document(file, request)

            results.append({
                "filename": file.filename,
                "document_id": document.id,
                "status": "success",
                "message": "Документ загружен"
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e)
            })

    return {
        "results": results,
        "total_files": len(files),
        "successful": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] == "error"])
    }
