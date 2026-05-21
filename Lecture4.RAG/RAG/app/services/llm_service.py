import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from abc import ABC, abstractmethod
import logging
import json

# OpenAI
try:
    import openai
except ImportError:
    openai = None

# Anthropic
try:
    import anthropic
except ImportError:
    anthropic = None

from app.config import settings

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Базовый класс для провайдеров LLM"""

    @abstractmethod
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Генерирует ответ на основе сообщений"""
        pass

    @abstractmethod
    async def generate_streaming_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Генерирует потоковый ответ"""
        pass


class OpenAILLMProvider(BaseLLMProvider):
    """Провайдер LLM через OpenAI API"""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        if not openai:
            raise ImportError("Установите openai: pip install openai")

        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Генерирует ответ через OpenAI API"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Ошибка генерации ответа OpenAI: {e}")
            raise

    async def generate_streaming_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Генерирует потоковый ответ через OpenAI API"""
        try:
            loop = asyncio.get_event_loop()
            stream = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    **kwargs
                )
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Ошибка потокового ответа OpenAI: {e}")
            raise


class AnthropicLLMProvider(BaseLLMProvider):
    """Провайдер LLM через Anthropic API"""

    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        if not anthropic:
            raise ImportError("Установите anthropic: pip install anthropic")

        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    def _convert_messages(self, messages: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
        """Конвертирует OpenAI формат сообщений в Anthropic формат"""
        system_message = ""
        converted_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                converted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        return system_message, converted_messages

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Генерирует ответ через Anthropic API"""
        try:
            system_message, converted_messages = self._convert_messages(
                messages)

            response = await self.client.messages.create(
                model=self.model,
                messages=converted_messages,
                system=system_message,
                temperature=temperature,
                max_tokens=max_tokens or 4000,
                **kwargs
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Ошибка генерации ответа Anthropic: {e}")
            raise

    async def generate_streaming_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Генерирует потоковый ответ через Anthropic API"""
        try:
            system_message, converted_messages = self._convert_messages(
                messages)

            async with self.client.messages.stream(
                model=self.model,
                messages=converted_messages,
                system=system_message,
                temperature=temperature,
                max_tokens=max_tokens or 4000,
                **kwargs
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error(f"Ошибка потокового ответа Anthropic: {e}")
            raise


class LLMService:
    """Сервис для работы с LLM"""

    def __init__(self, provider: Optional[BaseLLMProvider] = None):
        self.default_provider = provider or self._create_default_provider()

        # Системный промпт для RAG
        self.system_prompt = """Ты - полезный AI-ассистент, который отвечает на вопросы на основе предоставленной документации.

ВАЖНЫЕ ПРАВИЛА:
1. Давай точные и полезные ответы
2. Цитируй релевантные части документов когда это уместно
3. Отвечай на том же языке, на котором задан вопрос

Контекст из документов:
{context}

Ответь на следующий вопрос, используя только информацию из предоставленного контекста."""

    def _create_default_provider(self) -> BaseLLMProvider:
        """Создает провайдер по умолчанию"""
        if settings.DEFAULT_LLM_PROVIDER == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY не установлен")
            return OpenAILLMProvider(
                api_key=settings.OPENAI_API_KEY,
                model=settings.DEFAULT_LLM_MODEL
            )
        elif settings.DEFAULT_LLM_PROVIDER == "anthropic":
            if not settings.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY не установлен")
            return AnthropicLLMProvider(
                api_key=settings.ANTHROPIC_API_KEY,
                model=settings.DEFAULT_LLM_MODEL
            )
        else:
            raise ValueError(
                f"Неподдерживаемый провайдер LLM: {settings.DEFAULT_LLM_PROVIDER}")

    def _get_provider(self, llm_provider: Optional[str] = None, model_name: Optional[str] = None) -> BaseLLMProvider:
        """Получает провайдер LLM для конкретного запроса"""
        # Если провайдер не указан, используем по умолчанию
        if not llm_provider:
            return self.default_provider

        # Создаем провайдер для конкретного запроса
        if llm_provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY не установлен для OpenAI")
            return OpenAILLMProvider(
                api_key=settings.OPENAI_API_KEY,
                model=model_name or "gpt-4o-mini"
            )
        elif llm_provider == "anthropic":
            if not settings.ANTHROPIC_API_KEY:
                raise ValueError(
                    "ANTHROPIC_API_KEY не установлен для Anthropic")
            return AnthropicLLMProvider(
                api_key=settings.ANTHROPIC_API_KEY,
                model=model_name or "claude-3-sonnet-20240229"
            )
        else:
            raise ValueError(f"Неподдерживаемый провайдер: {llm_provider}")

    def _build_context(self, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Создает контекст из релевантных чанков"""
        if not relevant_chunks:
            return "Контекст не найден."

        context_parts = []
        for i, chunk in enumerate(relevant_chunks, 1):
            metadata = chunk.get('metadata', {})
            source = metadata.get('source', 'Неизвестный источник')

            context_parts.append(
                f"[Источник {i}: {source}]\n"
                f"{chunk['text']}\n"
            )

        return "\n".join(context_parts)

    def _prepare_messages(self, query: str, context: str) -> List[Dict[str, str]]:
        """Подготавливает сообщения для LLM"""
        system_message = self.system_prompt.format(context=context)

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
        ]

    async def generate_rag_response(
        self,
        query: str,
        relevant_chunks: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        llm_provider: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Генерирует RAG ответ на основе запроса и релевантных чанков

        Args:
            query: пользовательский запрос
            relevant_chunks: релевантные чанки документов
            temperature: температура генерации
            max_tokens: максимальное количество токенов
            llm_provider: провайдер LLM (openai, anthropic)
            model_name: конкретная модель

        Returns:
            Словарь с ответом и метаданными
        """
        try:
            # Получаем нужный провайдер
            provider = self._get_provider(llm_provider, model_name)

            # Строим контекст
            context = self._build_context(relevant_chunks)

            # Проверяем длину контекста
            if len(context) > settings.MAX_CONTEXT_LENGTH:
                # Обрезаем контекст если слишком длинный
                context = context[:settings.MAX_CONTEXT_LENGTH] + \
                    "...\n[Контекст обрезан]"

            # Подготавливаем сообщения
            messages = self._prepare_messages(query, context)

            # Генерируем ответ
            response = await provider.generate_response(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Определяем какая модель использовалась
            used_model = model_name or (
                provider.model if hasattr(provider, 'model')
                else f"{llm_provider}_model"
            )

            return {
                "answer": response,
                "sources_used": len(relevant_chunks),
                "context_length": len(context),
                "query": query,
                "llm_provider": llm_provider or settings.DEFAULT_LLM_PROVIDER,
                "model_used": used_model
            }

        except Exception as e:
            logger.error(f"Ошибка генерации RAG ответа: {e}")
            raise

    async def generate_rag_streaming_response(
        self,
        query: str,
        relevant_chunks: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Генерирует потоковый RAG ответ

        Yields:
            Словари с частями ответа и метаданными
        """
        try:
            # Строим контекст
            context = self._build_context(relevant_chunks)

            if len(context) > settings.MAX_CONTEXT_LENGTH:
                context = context[:settings.MAX_CONTEXT_LENGTH] + \
                    "...\n[Контекст обрезан]"

            # Подготавливаем сообщения
            messages = self._prepare_messages(query, context)

            # Отправляем метаданные в начале
            yield {
                "type": "metadata",
                "sources_used": len(relevant_chunks),
                "context_length": len(context),
                "query": query
            }

            # Генерируем потоковый ответ
            async for chunk in self.provider.generate_streaming_response(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            ):
                yield {
                    "type": "content",
                    "content": chunk
                }

        except Exception as e:
            logger.error(f"Ошибка потокового RAG ответа: {e}")
            yield {
                "type": "error",
                "error": str(e)
            }

    async def summarize_document(self, text: str, max_tokens: int = 500) -> str:
        """
        Создает краткое резюме документа

        Args:
            text: текст документа
            max_tokens: максимальная длина резюме

        Returns:
            Краткое резюме
        """
        try:
            # Обрезаем текст если слишком длинный
            if len(text) > settings.MAX_CONTEXT_LENGTH:
                text = text[:settings.MAX_CONTEXT_LENGTH] + "..."

            messages = [
                {
                    "role": "system",
                    "content": "Создай краткое и информативное резюме предоставленного текста. Выдели основные идеи и ключевые моменты."
                },
                {
                    "role": "user",
                    "content": f"Создай резюме этого текста:\n\n{text}"
                }
            ]

            summary = await self.default_provider.generate_response(
                messages=messages,
                temperature=0.3,
                max_tokens=max_tokens
            )

            return summary

        except Exception as e:
            logger.error(f"Ошибка создания резюме: {e}")
            return "Не удалось создать резюме документа."

    async def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Извлекает ключевые слова из текста

        Args:
            text: текст для анализа
            max_keywords: максимальное количество ключевых слов

        Returns:
            Список ключевых слов
        """
        try:
            if len(text) > settings.MAX_CONTEXT_LENGTH:
                text = text[:settings.MAX_CONTEXT_LENGTH] + "..."

            messages = [
                {
                    "role": "system",
                    "content": f"Извлеки {max_keywords} наиболее важных ключевых слов или фраз из текста. Верни их в виде JSON массива строк."
                },
                {
                    "role": "user",
                    "content": f"Текст для анализа:\n\n{text}"
                }
            ]

            response = await self.default_provider.generate_response(
                messages=messages,
                temperature=0.1,
                max_tokens=200
            )

            # Пытаемся парсить JSON
            try:
                keywords = json.loads(response)
                if isinstance(keywords, list):
                    return keywords[:max_keywords]
            except json.JSONDecodeError:
                # Если не JSON, разбиваем по строкам
                lines = response.strip().split('\n')
                keywords = []
                for line in lines:
                    line = line.strip().strip('-').strip('*').strip()
                    if line:
                        keywords.append(line)
                return keywords[:max_keywords]

            return []

        except Exception as e:
            logger.error(f"Ошибка извлечения ключевых слов: {e}")
            return []

    async def test_connection(self) -> bool:
        """Тестирует подключение к LLM провайдеру"""
        try:
            test_messages = [
                {"role": "user", "content": "Привет! Ответь одним словом: работает"}
            ]
            response = await self.default_provider.generate_response(test_messages, max_tokens=10)
            return bool(response and response.strip())
        except Exception as e:
            logger.error(f"Ошибка тестирования LLM: {e}")
            return False


# Глобальный экземпляр сервиса
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Возвращает глобальный экземпляр LLM сервиса"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
