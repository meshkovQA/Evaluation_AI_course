import os
import shutil
import logging
import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
import asyncio
import json
from fastapi import UploadFile, HTTPException

from app.models.document import (
    Document, DocumentType, DocumentStatus, DocumentMetadata,
    DocumentChunk, DocumentUploadRequest, DocumentUpdateRequest
)
from app.utils.file_parser import FileParser
from app.utils.text_splitter import TextSplitter
from app.config import settings

logger = logging.getLogger(__name__)


class DocumentService:
    """Сервис для работы с документами"""

    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or settings.DOCUMENTS_STORAGE_PATH
        self.file_parser = FileParser()
        self.text_splitter = TextSplitter()

        # Создаем директорию для хранения, если не существует
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)

        # Хранилище документов в памяти (в реальном проекте - база данных)
        self.documents: Dict[str, Document] = {}

        # Восстанавливаем документы из метаданных
        self._restore_documents_from_metadata()
        
        logger.info(f"✅ DocumentService инициализирован: {len(self.documents)} документов в памяти")

    def _save_documents_metadata(self):
        """Сохраняет метаданные документов в JSON файл"""
        metadata_file = os.path.join(self.storage_path, "documents_metadata.json")
        
        try:
            metadata = {}
            for doc_id, doc in self.documents.items():
                metadata[doc_id] = {
                    "id": doc.id,
                    "title": doc.title,
                    "file_path": doc.file_path,
                    "document_type": doc.document_type.value,
                    "status": doc.status.value,
                    "description": doc.description,
                    "tags": doc.tags,
                    "content": doc.content,  # Сохраняем содержимое
                    "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else None,
                    "processed_at": doc.processed_at.isoformat() if doc.processed_at else None,
                    "error_message": doc.error_message,
                    "metadata": {
                        "filename": doc.metadata.filename,
                        "file_size": doc.metadata.file_size,
                        "file_extension": doc.metadata.file_extension,
                        "word_count": doc.metadata.word_count,
                        "char_count": doc.metadata.char_count,
                        "created_at": doc.metadata.created_at,
                        "modified_at": doc.metadata.modified_at,
                    },
                    "chunks_count": len(doc.chunks)
                }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                
            logger.info(f"💾 Сохранено метаданных документов: {len(metadata)}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения метаданных: {e}")


    def _load_documents_metadata(self) -> Dict[str, Any]:
        """Загружает метаданные документов из JSON файла"""
        metadata_file = os.path.join(self.storage_path, "documents_metadata.json")
        
        if not os.path.exists(metadata_file):
            logger.info("📄 Файл метаданных не найден, начинаем с пустого хранилища")
            return {}
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.info(f"📂 Загружено метаданных: {len(metadata)} документов")
            return metadata
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки метаданных: {e}")
            return {}


    def _restore_documents_from_metadata(self):
        """
        Восстанавливает документы из сохраненных метаданных
        Чанки НЕ восстанавливаются в памяти, но остаются в ChromaDB для поиска
        """
        try:
            metadata = self._load_documents_metadata()
            
            if not metadata:
                logger.info("ℹ️  Нет сохраненных документов для восстановления")
                return
            
            restored_count = 0
            for doc_id, doc_meta in metadata.items():
                try:
                    # Проверяем, что файл еще существует
                    if not os.path.exists(doc_meta['file_path']):
                        logger.warning(f"⚠️  Файл не найден: {doc_meta['file_path']}, пропускаем документ {doc_id}")
                        continue
                    
                    # Создаем объект метаданных
                    file_metadata = DocumentMetadata(
                        filename=doc_meta['metadata']['filename'],
                        file_size=doc_meta['metadata']['file_size'],
                        file_extension=doc_meta['metadata']['file_extension'],
                        created_at=doc_meta['metadata'].get('created_at'),
                        modified_at=doc_meta['metadata'].get('modified_at'),
                        word_count=doc_meta['metadata'].get('word_count', 0),
                        char_count=doc_meta['metadata'].get('char_count', 0),
                    )
                    
                    # Парсим даты
                    uploaded_at = None
                    processed_at = None
                    if doc_meta.get('uploaded_at'):
                        uploaded_at = datetime.fromisoformat(doc_meta['uploaded_at'])
                    if doc_meta.get('processed_at'):
                        processed_at = datetime.fromisoformat(doc_meta['processed_at'])
                    
                    # Создаем документ БЕЗ чанков (они в ChromaDB)
                    document = Document(
                        id=doc_id,
                        title=doc_meta['title'],
                        file_path=doc_meta['file_path'],
                        document_type=DocumentType(doc_meta['document_type']),
                        status=DocumentStatus(doc_meta['status']),
                        metadata=file_metadata,
                        description=doc_meta.get('description'),
                        tags=doc_meta.get('tags', []),
                        content=doc_meta.get('content', ''),
                        uploaded_at=uploaded_at,
                        processed_at=processed_at,
                        error_message=doc_meta.get('error_message'),
                    )
                    
                    # Важно: chunks остается пустым в памяти, но данные есть в ChromaDB
                    document.chunks = []
                    
                    # Добавляем в память
                    self.documents[doc_id] = document
                    restored_count += 1
                    
                    logger.debug(f"✅ Восстановлен: {doc_meta['title']} (было {doc_meta['chunks_count']} чанков)")
                    
                except Exception as e:
                    logger.error(f"❌ Ошибка восстановления документа {doc_id}: {e}")
            
            logger.info(f"✅ Восстановлено документов: {restored_count}/{len(metadata)}")
            logger.info("💡 Чанки НЕ загружены в память, но доступны для поиска в ChromaDB")
            
        except Exception as e:
            logger.error(f"❌ Ошибка восстановления документов: {e}")

    def _generate_stable_id(self, file_path: str, filename: str) -> str:
        """Генерирует стабильный ID на основе файла"""
        # Используем имя файла + размер + время модификации для создания стабильного хэша
        file_stat = os.stat(file_path)
        hash_input = f"{filename}_{file_stat.st_size}_{file_stat.st_mtime}"

        # Создаем MD5 хэш и берем первые 8 символов
        hash_object = hashlib.md5(hash_input.encode())
        short_hash = hash_object.hexdigest()[:8]

        return f"doc_{short_hash}"

    async def upload_document(
        self,
        file: UploadFile,
        request: DocumentUploadRequest
    ) -> Document:
        """
        Загружает и обрабатывает документ

        Args:
            file: загружаемый файл
            request: дополнительные параметры

        Returns:
            Обработанный документ
        """
        # Проверяем поддерживаемость формата
        if not self.file_parser.is_supported(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Неподдерживаемый формат файла: {file.filename}"
            )

        # Генерируем уникальное имя файла
        file_extension = Path(file.filename).suffix
        safe_filename = self._generate_safe_filename(file.filename)
        file_path = os.path.join(self.storage_path, safe_filename)

        try:
            # Сохраняем файл
            await self._save_uploaded_file(file, file_path)

            # Создаем базовые метаданные
            metadata = self._create_metadata(file_path, file.filename)

            # Генерируем стабильный ID
            stable_id = self._generate_stable_id(file_path, safe_filename)

            # Создаем документ со стабильным ID
            document = Document(
                id=stable_id,  # Используем стабильный ID
                title=request.title or Path(file.filename).stem,
                file_path=file_path,
                document_type=DocumentType(
                    self.file_parser.get_file_type(file_path)),
                status=DocumentStatus.PROCESSING,
                metadata=metadata,
                description=request.description,
                tags=request.tags
            )

            # Сохраняем в хранилище
            self.documents[document.id] = document

                        # Запускаем обработку асинхронно
            asyncio.create_task(self._process_document(document.id))

            # Сохраняем метаданные
            self._save_documents_metadata()

            return document

        except Exception as e:
            # Удаляем файл в случае ошибки
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка при загрузке файла: {str(e)}"
            )

    async def _process_document(self, document_id: str):
        """Асинхронная обработка документа с детальным логированием"""
        document = self.documents.get(document_id)
        if not document:
            logger.error(f"❌ Документ {document_id} не найден для обработки")
            return

        logger.info(f"🔄 Начало обработки документа: {document.title} (ID: {document_id})")
        
        try:
            # 1. Парсим содержимое файла
            logger.info(f"📄 Парсинг файла: {document.file_path}")
            parse_result = self.file_parser.parse_file(document.file_path)

            if not parse_result['success']:
                logger.error(f"❌ Ошибка парсинга: {parse_result['error']}")
                document.status = DocumentStatus.ERROR
                document.error_message = parse_result['error']
                return

            # 2. Обновляем содержимое документа
            document.content = parse_result['text']
            logger.info(f"✅ Текст извлечен: {len(document.content)} символов")

            if not document.content or len(document.content.strip()) < 10:
                logger.error(f"❌ Документ пустой или слишком короткий")
                document.status = DocumentStatus.ERROR
                document.error_message = "Документ пустой или содержит слишком мало текста"
                return

            # 3. Обновляем метаданные
            document.metadata.word_count = len(document.content.split())
            document.metadata.char_count = len(document.content)
            logger.info(f"📊 Статистика: {document.metadata.word_count} слов, {document.metadata.char_count} символов")

            # 4. Создаем чанки для векторного поиска
            logger.info(f"✂️  Разделение на чанки...")
            chunks_text = self.text_splitter.split_text(document.content)
            logger.info(f"✅ Создано {len(chunks_text)} чанков")
            
            if not chunks_text:
                logger.error(f"❌ Не удалось создать чанки")
                document.status = DocumentStatus.ERROR
                document.error_message = "Не удалось разделить текст на чанки"
                return

            # Создаем объекты чанков
            document.chunks = [
                DocumentChunk(
                    document_id=document.id,
                    chunk_index=i,
                    text=chunk,
                    metadata={
                        'source': document.title,
                        'document_type': document.document_type.value,
                        'chunk_size': len(chunk)
                    }
                )
                for i, chunk in enumerate(chunks_text)
            ]
            logger.info(f"📦 Создано {len(document.chunks)} объектов DocumentChunk")

            # 5. Создаем embeddings для чанков
            logger.info(f"🧮 Создание embeddings...")
            try:
                await self._create_embeddings_for_chunks(document)
                
                # Проверяем, что embeddings созданы
                chunks_with_embeddings = [c for c in document.chunks if c.embedding is not None]
                logger.info(f"✅ Embeddings созданы для {len(chunks_with_embeddings)}/{len(document.chunks)} чанков")
                
                if len(chunks_with_embeddings) == 0:
                    logger.error(f"❌ Не создано ни одного embedding")
                    document.status = DocumentStatus.ERROR
                    document.error_message = "Не удалось создать embeddings"
                    return
                    
            except Exception as e:
                logger.error(f"❌ Ошибка создания embeddings: {e}", exc_info=True)
                document.status = DocumentStatus.ERROR
                document.error_message = f"Ошибка создания embeddings: {str(e)}"
                return

            # 6. Сохраняем в векторное хранилище
            logger.info(f"💾 Сохранение в векторное хранилище...")
            try:
                await self._save_to_vector_store(document)
                logger.info(f"✅ Чанки сохранены в векторное хранилище")
            except Exception as e:
                logger.error(f"❌ Ошибка сохранения в векторное хранилище: {e}", exc_info=True)
                # Не останавливаем обработку, но логируем ошибку
                document.error_message = f"Предупреждение: ошибка сохранения в векторное хранилище: {str(e)}"

            # 7. Помечаем как готовый
            document.status = DocumentStatus.READY
            document.processed_at = datetime.utcnow()
            logger.info(f"✅ Документ {document.title} успешно обработан!")
            logger.info(f"📊 Итого: {len(document.chunks)} чанков, статус: {document.status.value}")

            self._save_documents_metadata()

        except Exception as e:
            logger.error(f"❌ Критическая ошибка обработки документа {document_id}: {e}", exc_info=True)
            document.status = DocumentStatus.ERROR
            document.error_message = f"Критическая ошибка: {str(e)}"
            
            # Сохраняем метаданные даже при ошибке
            self._save_documents_metadata()

    async def _create_embeddings_for_chunks(self, document: Document):
        """Создает embeddings для чанков документа"""
        try:
            logger.info(f"🔧 Инициализация embedding сервиса...")
            from app.services.embedding_service import get_embedding_service

            embedding_service = get_embedding_service()
            logger.info(f"✅ Embedding сервис инициализирован")

            # Извлекаем тексты чанков
            chunk_texts = [chunk.text for chunk in document.chunks]
            logger.info(f"📝 Подготовлено {len(chunk_texts)} текстов для embedding")

            if chunk_texts:
                # Создаем embeddings
                logger.info(f"🚀 Отправка запроса на создание embeddings...")
                embeddings = await embedding_service.create_embeddings_for_chunks(chunk_texts)
                logger.info(f"✅ Получено {len(embeddings)} embeddings")

                # Присваиваем embeddings чанкам
                for i, (chunk, embedding) in enumerate(zip(document.chunks, embeddings)):
                    chunk.embedding = embedding
                    logger.debug(f"   Чанк {i}: embedding размер {len(embedding)}")
                    
                logger.info(f"✅ Embeddings присвоены всем чанкам")

        except ImportError as e:
            logger.error(f"❌ Ошибка импорта embedding сервиса: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Ошибка создания embeddings для документа {document.id}: {e}", exc_info=True)
            raise

    async def _save_to_vector_store(self, document: Document):
        """Сохраняет чанки документа в векторное хранилище"""
        try:
            logger.info(f"🔧 Инициализация векторного хранилища...")
            from app.services.vector_store import get_vector_store_service

            vector_store = get_vector_store_service()
            logger.info(f"✅ Векторное хранилище инициализировано")

            # Фильтруем чанки с embeddings
            chunks_with_embeddings = [
                chunk for chunk in document.chunks
                if chunk.embedding is not None
            ]
            
            logger.info(f"📦 Чанков с embeddings: {len(chunks_with_embeddings)}/{len(document.chunks)}")

            if chunks_with_embeddings:
                logger.info(f"💾 Сохранение {len(chunks_with_embeddings)} чанков в векторное хранилище...")
                success = await vector_store.add_document_chunks(chunks_with_embeddings)
                
                if not success:
                    logger.warning(f"⚠️  Не удалось сохранить чанки документа {document.id} в векторное хранилище")
                    raise Exception("Векторное хранилище вернуло success=False")
                else:
                    logger.info(f"✅ Чанки успешно сохранены в векторное хранилище")

        except ImportError as e:
            logger.error(f"❌ Ошибка импорта векторного хранилища: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения в векторное хранилище для документа {document.id}: {e}", exc_info=True)
            raise

        

    def get_document(self, document_id: str) -> Optional[Document]:
        """Получает документ по ID"""
        return self.documents.get(document_id)

    def get_documents(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[DocumentStatus] = None,
        document_type: Optional[DocumentType] = None
    ) -> Dict[str, Any]:
        """
        Получает список документов с пагинацией и фильтрацией

        Returns:
            Словарь с документами и метаданными пагинации
        """
        documents = list(self.documents.values())

        # Фильтрация
        if status:
            documents = [doc for doc in documents if doc.status == status]

        if document_type:
            documents = [
                doc for doc in documents if doc.document_type == document_type]

        # Сортировка по дате загрузки (новые первыми)
        documents.sort(key=lambda x: x.uploaded_at, reverse=True)

        # Пагинация
        total = len(documents)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_documents = documents[start_idx:end_idx]

        return {
            'documents': page_documents,
            'total': total,
            'page': page,
            'page_size': page_size,
            'total_pages': (total + page_size - 1) // page_size
        }

    def update_document(
        self,
        document_id: str,
        request: DocumentUpdateRequest
    ) -> Optional[Document]:
        """Обновляет метаданные документа"""
        document = self.documents.get(document_id)
        if not document:
            return None

        if request.title is not None:
            document.title = request.title
        if request.description is not None:
            document.description = request.description
        if request.tags is not None:
            document.tags = request.tags

        self._save_documents_metadata()

        return document

    def delete_document(self, document_id: str) -> bool:
        """Удаляет документ и связанный файл"""
        document = self.documents.get(document_id)
        if not document:
            return False

        try:
            # Удаляем из векторного хранилища
            asyncio.create_task(self._delete_from_vector_store(document_id))

            # Удаляем файл
            if os.path.exists(document.file_path):
                os.remove(document.file_path)

            # Удаляем из хранилища
            del self.documents[document_id]

            self._save_documents_metadata()
            return True

        except Exception:
            return False

    async def _delete_from_vector_store(self, document_id: str):
        """Удаляет документ из векторного хранилища"""
        try:
            from app.services.vector_store import get_vector_store_service

            vector_store = get_vector_store_service()
            await vector_store.delete_document_chunks(document_id)

        except Exception as e:
            logger.error(f"Ошибка удаления из векторного хранилища: {e}")

    def search_documents(self, query: str) -> List[Document]:
        """Простой поиск документов по названию и содержимому"""
        query_lower = query.lower()
        results = []

        for document in self.documents.values():
            if (query_lower in document.title.lower() or
                query_lower in document.content.lower() or
                    any(query_lower in tag.lower() for tag in document.tags)):
                results.append(document)

        return results

    async def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Получает чанки: сначала из памяти, потом из ChromaDB"""
        document = self.documents.get(document_id)
        if not document:
            return []
        
        # Если чанки в памяти - возвращаем их
        if document.chunks:
            return [
                {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "has_embedding": chunk.embedding is not None
                }
                for chunk in document.chunks
            ]
        
        # Иначе загружаем из ChromaDB
        logger.info(f"📂 Загружаем чанки из ChromaDB для {document_id}")
        
        try:
            from app.services.vector_store import get_vector_store_service
            vector_store = get_vector_store_service()
            chunks = await vector_store.get_document_chunks_from_store(document_id)
            logger.info(f"✅ Загружено {len(chunks)} чанков")
            return chunks
        except Exception as e:
            logger.error(f"❌ Ошибка: {e}", exc_info=True)
            return []

    async def _save_uploaded_file(self, file: UploadFile, file_path: str):
        """Сохраняет загруженный файл на диск"""
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    def _generate_safe_filename(self, filename: str) -> str:
        """Генерирует безопасное имя файла"""
        # Получаем расширение
        extension = Path(filename).suffix
        # Создаем уникальное имя
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = f"{timestamp}_{filename.replace(' ', '_')}"
        return safe_name

    def _create_metadata(self, file_path: str, original_filename: str) -> DocumentMetadata:
        """Создает метаданные для документа"""
        file_stat = os.stat(file_path)

        return DocumentMetadata(
            filename=original_filename,
            file_size=file_stat.st_size,
            file_extension=Path(file_path).suffix,
            created_at=file_stat.st_ctime,
            modified_at=file_stat.st_mtime
        )

# Глобальный экземпляр сервиса документов (синглтон)
_document_service_instance: Optional[DocumentService] = None


def get_document_service() -> DocumentService:
    """
    Возвращает глобальный экземпляр сервиса документов.
    Гарантирует, что фоновые задачи обработки не теряются.
    """
    global _document_service_instance
    if _document_service_instance is None:
        _document_service_instance = DocumentService()
        logger.info("✅ Создан глобальный экземпляр DocumentService")
    return _document_service_instance