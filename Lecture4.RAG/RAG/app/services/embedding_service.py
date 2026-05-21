import asyncio
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
import numpy as np
import logging

# OpenAI
try:
    import openai
except ImportError:
    openai = None

# HuggingFace
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from app.config import settings

logger = logging.getLogger(__name__)


class BaseEmbeddingProvider(ABC):
    """Базовый класс для провайдеров embeddings"""

    @abstractmethod
    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Создает embeddings для списка текстов"""
        pass

    @abstractmethod
    async def create_embedding(self, text: str) -> List[float]:
        """Создает embedding для одного текста"""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Размерность векторов"""
        pass


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Провайдер embeddings через OpenAI API"""

    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        if not openai:
            raise ImportError("Установите openai: pip install openai")

        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self._dimension = 1536 if "ada-002" in model else 1536

    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Создает embeddings для списка текстов"""
        try:
            # OpenAI API поддерживает батчи до 2048 текстов
            batch_size = 100
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                # Выполняем запрос в отдельном потоке
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.embeddings.create(
                        input=batch,
                        model=self.model
                    )
                )

                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)

                # Небольшая задержка между батчами
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)

            return all_embeddings

        except Exception as e:
            logger.error(f"Ошибка создания embeddings через OpenAI: {e}")
            raise

    async def create_embedding(self, text: str) -> List[float]:
        """Создает embedding для одного текста"""
        embeddings = await self.create_embeddings([text])
        return embeddings[0]

    @property
    def dimension(self) -> int:
        return self._dimension


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """Провайдер embeddings через HuggingFace models"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if not SentenceTransformer:
            raise ImportError(
                "Установите sentence-transformers: pip install sentence-transformers")

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()

    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Создает embeddings для списка текстов"""
        try:
            # Выполняем в отдельном потоке так как модель работает синхронно
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(texts, convert_to_numpy=True)
            )

            return embeddings.tolist()

        except Exception as e:
            logger.error(f"Ошибка создания embeddings через HuggingFace: {e}")
            raise

    async def create_embedding(self, text: str) -> List[float]:
        """Создает embedding для одного текста"""
        embeddings = await self.create_embeddings([text])
        return embeddings[0]

    @property
    def dimension(self) -> int:
        return self._dimension


class EmbeddingService:
    """Сервис для работы с embeddings"""

    def __init__(self, provider: Optional[BaseEmbeddingProvider] = None):
        self.provider = provider or self._create_default_provider()

    def _create_default_provider(self) -> BaseEmbeddingProvider:
        """Создает провайдер по умолчанию на основе настроек"""
        if settings.EMBEDDING_PROVIDER == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY не установлен")
            return OpenAIEmbeddingProvider(
                api_key=settings.OPENAI_API_KEY,
                model=settings.EMBEDDING_MODEL
            )
        elif settings.EMBEDDING_PROVIDER == "huggingface":
            return HuggingFaceEmbeddingProvider(
                model_name=settings.EMBEDDING_MODEL
            )
        else:
            raise ValueError(
                f"Неподдерживаемый провайдер: {settings.EMBEDDING_PROVIDER}")

    async def create_embeddings_for_chunks(self, chunks: List[str]) -> List[List[float]]:
        """
        Создает embeddings для списка чанков

        Args:
            chunks: список текстовых чанков

        Returns:
            Список векторов embeddings
        """
        if not chunks:
            return []

        logger.info(f"Создание embeddings для {len(chunks)} чанков")

        try:
            embeddings = await self.provider.create_embeddings(chunks)
            logger.info(f"Успешно создано {len(embeddings)} embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Ошибка создания embeddings: {e}")
            raise

    async def create_embedding_for_query(self, query: str) -> List[float]:
        """
        Создает embedding для поискового запроса

        Args:
            query: поисковый запрос

        Returns:
            Вектор embedding
        """
        logger.debug(f"Создание embedding для запроса: {query[:100]}...")

        try:
            embedding = await self.provider.create_embedding(query)
            return embedding

        except Exception as e:
            logger.error(f"Ошибка создания embedding для запроса: {e}")
            raise

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Вычисляет косинусное сходство между двумя векторами

        Args:
            embedding1: первый вектор
            embedding2: второй вектор

        Returns:
            Значение сходства от 0 до 1
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Косинусное сходство
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(max(0, similarity))  # Обеспечиваем значение от 0 до 1

    def find_most_similar(
        self,
        query_embedding: List[float],
        embeddings: List[List[float]],
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Находит наиболее похожие embeddings

        Args:
            query_embedding: embedding запроса
            embeddings: список embeddings для сравнения
            top_k: количество результатов
            threshold: минимальный порог сходства

        Returns:
            Список словарей с индексами и оценками сходства
        """
        similarities = []

        for i, embedding in enumerate(embeddings):
            similarity = self.calculate_similarity(query_embedding, embedding)
            if similarity >= threshold:
                similarities.append({
                    'index': i,
                    'similarity': similarity
                })

        # Сортируем по убыванию сходства
        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        return similarities[:top_k]

    @property
    def embedding_dimension(self) -> int:
        """Возвращает размерность векторов"""
        return self.provider.dimension

    async def test_connection(self) -> bool:
        """Тестирует подключение к провайдеру embeddings"""
        try:
            test_embedding = await self.provider.create_embedding("test")
            return len(test_embedding) == self.embedding_dimension
        except Exception as e:
            logger.error(f"Ошибка тестирования подключения: {e}")
            return False


# Функция для создания глобального экземпляра сервиса
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Возвращает глобальный экземпляр сервиса embeddings"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
