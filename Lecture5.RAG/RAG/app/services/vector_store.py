import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import logging
from pathlib import Path

# ChromaDB
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None

# Pinecone
try:
    import pinecone
except ImportError:
    pinecone = None

# FAISS
try:
    import faiss
    import numpy as np
except ImportError:
    faiss = None
    np = None

from app.config import settings
from app.models.document import DocumentChunk

logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """Базовый класс для векторных хранилищ"""

    @abstractmethod
    async def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """Добавляет документы в векторное хранилище"""
        pass

    @abstractmethod
    async def search(self, query_embedding: List[float], top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Выполняет поиск по векторному хранилищу"""
        pass

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Удаляет все чанки документа"""
        pass

    @abstractmethod
    async def get_collection_info(self) -> Dict[str, Any]:
        """Возвращает информацию о коллекции"""
        pass


class ChromaVectorStore(BaseVectorStore):
    """Векторное хранилище на основе ChromaDB"""

    def __init__(self, persist_directory: str = None, collection_name: str = "rag_documents"):
        if not chromadb:
            raise ImportError("Установите chromadb: pip install chromadb")

        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIRECTORY
        self.collection_name = collection_name

        # Создаем директорию если не существует
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Инициализируем клиент
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # Получаем или создаем коллекцию
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "RAG documents collection"}
        )

        logger.info(f"ChromaDB инициализирован: {self.persist_directory}")

    async def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """Добавляет чанки документов в ChromaDB"""
        if not chunks:
            return True

        try:
            # Подготавливаем данные для ChromaDB
            ids = []
            embeddings = []
            documents = []
            metadatas = []

            for chunk in chunks:
                if chunk.embedding is None:
                    logger.warning(f"Чанк {chunk.id} не имеет embedding")
                    continue

                ids.append(chunk.id)
                embeddings.append(chunk.embedding)
                documents.append(chunk.text)
                metadatas.append({
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata
                })

            if not ids:
                logger.warning("Нет чанков с embeddings для добавления")
                return False

            # Добавляем в коллекцию
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
            )

            logger.info(f"Добавлено {len(ids)} чанков в ChromaDB")
            return True

        except Exception as e:
            logger.error(f"Ошибка добавления в ChromaDB: {e}")
            return False

    async def search(self, query_embedding: List[float], top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Выполняет поиск в ChromaDB"""
        try:
            # Проверяем количество документов в коллекции
            total_count = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.count()
            )
            logger.info(f"🔍 Поиск в ChromaDB: всего {total_count} чанков в коллекции")
            
            if total_count == 0:
                logger.warning("⚠️  ChromaDB коллекция ПУСТАЯ!")
                return []
            
            # Выполняем поиск
            logger.info(f"🔎 Запрос поиска: top_k={top_k}, embedding_dim={len(query_embedding)}")
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, total_count),
                    include=["documents", "metadatas", "distances"]
                )
            )

            # Преобразуем результаты
            search_results = []
            if results['ids'] and results['ids'][0]:
                logger.info(f"📊 ChromaDB вернула {len(results['ids'][0])} результатов")
                
                for i in range(len(results['ids'][0])):
                    distance = results['distances'][0][i]
                    similarity = 1 - distance
                    
                    result = {
                        "chunk_id": results['ids'][0][i],
                        "text": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": distance,
                        "similarity": similarity
                    }
                    
                    logger.info(f"   {i+1}. similarity={similarity:.4f}, distance={distance:.4f}, doc={results['metadatas'][0][i].get('source', 'Unknown')}")
                    
                    search_results.append(result)
            else:
                logger.warning("⚠️  ChromaDB вернула 0 результатов")

            return search_results

        except Exception as e:
            logger.error(f"❌ Ошибка поиска в ChromaDB: {e}", exc_info=True)
            return []

    async def delete_document(self, document_id: str) -> bool:
        """Удаляет все чанки документа из ChromaDB"""
        try:
            # Ищем все чанки документа
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.get(
                    where={"document_id": document_id},
                    include=["metadatas"]
                )
            )

            if results['ids']:
                # Удаляем найденные чанки
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.collection.delete(ids=results['ids'])
                )
                logger.info(
                    f"Удалено {len(results['ids'])} чанков документа {document_id}")

            return True

        except Exception as e:
            logger.error(f"Ошибка удаления документа из ChromaDB: {e}")
            return False

    async def get_collection_info(self) -> Dict[str, Any]:
        """Возвращает информацию о коллекции ChromaDB"""
        try:
            count = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.count()
            )

            return {
                "type": "chromadb",
                "collection_name": self.collection_name,
                "total_chunks": count,
                "persist_directory": self.persist_directory
            }

        except Exception as e:
            logger.error(f"Ошибка получения информации о коллекции: {e}")
            return {"type": "chromadb", "error": str(e)}
    
    
    async def get_document_chunks_from_db(self, document_id: str) -> List[Dict[str, Any]]:
        """Получает все чанки документа из ChromaDB"""
        try:
            logger.info(f"📂 Получение чанков документа {document_id} из ChromaDB")
            
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.get(
                    where={"document_id": document_id},
                    include=["documents", "metadatas"]
                )
            )
            
            chunks = []
            if results['ids']:
                logger.info(f"✅ Найдено {len(results['ids'])} чанков")
                
                for i, chunk_id in enumerate(results['ids']):
                    chunks.append({
                        "id": chunk_id,
                        "document_id": document_id,
                        "chunk_index": results['metadatas'][i].get('chunk_index', i),
                        "text": results['documents'][i],
                        "metadata": results['metadatas'][i],
                        "has_embedding": True
                    })
                
                chunks.sort(key=lambda x: x.get('chunk_index', 0))
            
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Ошибка: {e}", exc_info=True)
            return []


class PineconeVectorStore(BaseVectorStore):
    """Векторное хранилище на основе Pinecone"""

    def __init__(self, api_key: str = None, environment: str = None, index_name: str = None):
        if not pinecone:
            raise ImportError(
                "Установите pinecone: pip install pinecone-client")

        self.api_key = api_key or settings.PINECONE_API_KEY
        self.environment = environment or settings.PINECONE_ENVIRONMENT
        self.index_name = index_name or settings.PINECONE_INDEX_NAME

        if not all([self.api_key, self.environment, self.index_name]):
            raise ValueError(
                "Необходимо указать PINECONE_API_KEY, PINECONE_ENVIRONMENT и PINECONE_INDEX_NAME")

        # Инициализируем Pinecone
        pinecone.init(
            api_key=self.api_key,
            environment=self.environment
        )

        self.index = pinecone.Index(self.index_name)
        logger.info(f"Pinecone инициализирован: {self.index_name}")

    async def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """Добавляет чанки в Pinecone"""
        if not chunks:
            return True

        try:
            vectors = []
            for chunk in chunks:
                if chunk.embedding is None:
                    continue

                vectors.append({
                    "id": chunk.id,
                    "values": chunk.embedding,
                    "metadata": {
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        **chunk.metadata
                    }
                })

            if vectors:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.index.upsert(vectors=vectors)
                )
                logger.info(f"Добавлено {len(vectors)} чанков в Pinecone")

            return True

        except Exception as e:
            logger.error(f"Ошибка добавления в Pinecone: {e}")
            return False

    async def search(self, query_embedding: List[float], top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Выполняет поиск в Pinecone"""
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True
                )
            )

            search_results = []
            for match in results['matches']:
                search_results.append({
                    "chunk_id": match['id'],
                    "text": match['metadata'].get('text', ''),
                    "metadata": match['metadata'],
                    "similarity": match['score']
                })

            return search_results

        except Exception as e:
            logger.error(f"Ошибка поиска в Pinecone: {e}")
            return []

    async def delete_document(self, document_id: str) -> bool:
        """Удаляет документ из Pinecone"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.index.delete(filter={"document_id": document_id})
            )
            return True

        except Exception as e:
            logger.error(f"Ошибка удаления из Pinecone: {e}")
            return False

    async def get_collection_info(self) -> Dict[str, Any]:
        """Возвращает информацию об индексе Pinecone"""
        try:
            stats = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.index.describe_index_stats()
            )

            return {
                "type": "pinecone",
                "index_name": self.index_name,
                "total_vectors": stats.get('total_vector_count', 0),
                "dimension": stats.get('dimension', 0)
            }

        except Exception as e:
            logger.error(f"Ошибка получения статистики Pinecone: {e}")
            return {"type": "pinecone", "error": str(e)}


class VectorStoreService:
    """Сервис для работы с векторными хранилищами"""

    def __init__(self, store: Optional[BaseVectorStore] = None):
        self.store = store or self._create_default_store()

    def _create_default_store(self) -> BaseVectorStore:
        """Создает векторное хранилище по умолчанию"""
        if settings.VECTOR_DB_TYPE == "chroma":
            return ChromaVectorStore()
        elif settings.VECTOR_DB_TYPE == "pinecone":
            return PineconeVectorStore()
        else:
            raise ValueError(
                f"Неподдерживаемый тип векторной БД: {settings.VECTOR_DB_TYPE}")

    async def add_document_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Добавляет чанки документа в векторное хранилище"""
        return await self.store.add_documents(chunks)

    async def search_similar_chunks(
        self,
        query_embedding: List[float],
        top_k: int = None,
        similarity_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Ищет похожие чанки

        Args:
            query_embedding: embedding запроса
            top_k: количество результатов
            similarity_threshold: минимальный порог сходства

        Returns:
            Список найденных чанков с метаданными
        """
        top_k = top_k or settings.MAX_RELEVANT_CHUNKS
        similarity_threshold = similarity_threshold or settings.SIMILARITY_THRESHOLD

        results = await self.store.search(query_embedding, top_k)

        # Фильтруем по порогу сходства
        filtered_results = []
        for result in results:
            similarity = result.get('similarity', 0)
            if similarity >= similarity_threshold:
                filtered_results.append(result)

        return filtered_results

    async def delete_document_chunks(self, document_id: str) -> bool:
        """Удаляет все чанки документа"""
        return await self.store.delete_document(document_id)

    async def get_store_info(self) -> Dict[str, Any]:
        """Возвращает информацию о векторном хранилище"""
        return await self.store.get_collection_info()
    
    async def get_document_chunks_from_store(self, document_id: str) -> List[Dict[str, Any]]:
        """Получает чанки документа из векторного хранилища"""
        if isinstance(self.store, ChromaVectorStore):
            return await self.store.get_document_chunks_from_db(document_id)
        else:
            logger.warning(f"Не реализовано для {type(self.store).__name__}")
            return []

    async def test_connection(self) -> bool:
        """Тестирует подключение к векторному хранилищу"""
        try:
            info = await self.get_store_info()
            return not info.get('error')
        except Exception as e:
            logger.error(f"Ошибка тестирования векторного хранилища: {e}")
            return False


# Глобальный экземпляр сервиса
_vector_store_service: Optional[VectorStoreService] = None


def get_vector_store_service() -> VectorStoreService:
    """Возвращает глобальный экземпляр сервиса векторного хранилища"""
    global _vector_store_service
    if _vector_store_service is None:
        _vector_store_service = VectorStoreService()
    return _vector_store_service
