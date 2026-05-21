import re
from typing import List, Optional
from abc import ABC, abstractmethod


class BaseTextSplitter(ABC):
    """Базовый класс для разделения текста на чанки"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Args:
            chunk_size: размер чанка в символах
            chunk_overlap: перекрытие между чанками в символах
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Разделяет текст на чанки"""
        pass

    def _merge_splits(self, splits: List[str], separator: str = " ") -> List[str]:
        """Объединяет мелкие части в чанки нужного размера"""
        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_length = len(split)

            # Если добавление этого split превысит размер чанка
            if current_length + split_length > self.chunk_size and current_chunk:
                # Сохраняем текущий чанк
                chunk_text = separator.join(current_chunk)
                if chunk_text.strip():
                    chunks.append(chunk_text.strip())

                # Начинаем новый чанк с перекрытием
                if self.chunk_overlap > 0:
                    current_chunk = self._get_overlap_splits(
                        current_chunk, separator)
                    current_length = len(separator.join(current_chunk))
                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(split)
            current_length += split_length + len(separator)

        # Добавляем последний чанк
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())

        return chunks

    def _get_overlap_splits(self, splits: List[str], separator: str) -> List[str]:
        """Получает части для перекрытия между чанками"""
        overlap_length = 0
        overlap_splits = []

        # Берем splits с конца пока не достигнем нужного перекрытия
        for split in reversed(splits):
            if overlap_length + len(split) + len(separator) <= self.chunk_overlap:
                overlap_splits.insert(0, split)
                overlap_length += len(split) + len(separator)
            else:
                break

        return overlap_splits


class RecursiveCharacterTextSplitter(BaseTextSplitter):
    """
    Рекурсивно разделяет текст, пытаясь сохранить смысловые границы
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: Optional[List[str]] = None):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or [
            "\n\n", "\n", ".", "!", "?", ";", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Рекурсивно разделяет текст"""
        return self._split_text_recursive(text, self.separators)

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Рекурсивная функция разделения"""
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        # Если сепаратор пустой, делим по символам
        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)

        # Фильтруем пустые части
        good_splits = []
        for split in splits:
            if len(split) < self.chunk_size:
                good_splits.append(split)
            else:
                # Если часть слишком большая, рекурсивно разделяем дальше
                if remaining_separators:
                    sub_splits = self._split_text_recursive(
                        split, remaining_separators)
                    good_splits.extend(sub_splits)
                else:
                    good_splits.append(split)

        return self._merge_splits(good_splits, separator)


class SentenceTextSplitter(BaseTextSplitter):
    """
    Разделяет текст по предложениям
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        # Паттерн для разделения по предложениям
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')

    def split_text(self, text: str) -> List[str]:
        """Разделяет текст по предложениям"""
        sentences = self.sentence_pattern.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        return self._merge_splits(sentences, " ")


class ParagraphTextSplitter(BaseTextSplitter):
    """
    Разделяет текст по абзацам
    """

    def split_text(self, text: str) -> List[str]:
        """Разделяет текст по абзацам"""
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        return self._merge_splits(paragraphs, '\n\n')


class FixedSizeTextSplitter(BaseTextSplitter):
    """
    Разделяет текст на чанки фиксированного размера
    """

    def split_text(self, text: str) -> List[str]:
        """Разделяет текст на фиксированные чанки"""
        chunks = []

        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())

        return chunks


class MarkdownTextSplitter(BaseTextSplitter):
    """
    Специальный разделитель для Markdown документов
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        self.md_separators = [
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n##### ",
            "\n###### ",
            "\n\n",
            "\n",
            " ",
            ""
        ]

    def split_text(self, text: str) -> List[str]:
        """Разделяет Markdown текст сохраняя структуру"""
        return self._split_text_recursive(text, self.md_separators)

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Рекурсивное разделение с учетом Markdown структуры"""
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)

        # Объединяем части обратно с сепаратором (кроме последнего)
        formatted_splits = []
        for i, split in enumerate(splits):
            if i > 0 and separator != "":
                split = separator + split
            formatted_splits.append(split)

        good_splits = []
        for split in formatted_splits:
            if len(split) < self.chunk_size:
                good_splits.append(split)
            else:
                if remaining_separators:
                    sub_splits = self._split_text_recursive(
                        split, remaining_separators)
                    good_splits.extend(sub_splits)
                else:
                    good_splits.append(split)

        return self._merge_splits(good_splits, "")


class TextSplitter:
    """
    Фабрика для создания разделителей текста
    """

    def __init__(self,
                 splitter_type: str = "recursive",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Args:
            splitter_type: тип разделителя (recursive, sentence, paragraph, fixed, markdown)
            chunk_size: размер чанка
            chunk_overlap: перекрытие между чанками
        """
        self.splitter_type = splitter_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = self._create_splitter()

    def _create_splitter(self) -> BaseTextSplitter:
        """Создает нужный тип разделителя"""
        if self.splitter_type == "recursive":
            return RecursiveCharacterTextSplitter(self.chunk_size, self.chunk_overlap)
        elif self.splitter_type == "sentence":
            return SentenceTextSplitter(self.chunk_size, self.chunk_overlap)
        elif self.splitter_type == "paragraph":
            return ParagraphTextSplitter(self.chunk_size, self.chunk_overlap)
        elif self.splitter_type == "fixed":
            return FixedSizeTextSplitter(self.chunk_size, self.chunk_overlap)
        elif self.splitter_type == "markdown":
            return MarkdownTextSplitter(self.chunk_size, self.chunk_overlap)
        else:
            raise ValueError(
                f"Неподдерживаемый тип разделителя: {self.splitter_type}")

    def split_text(self, text: str) -> List[str]:
        """Разделяет текст на чанки"""
        return self.splitter.split_text(text)

    def get_chunks_info(self, text: str) -> dict:
        """Возвращает информацию о разделении текста"""
        chunks = self.split_text(text)

        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0,
            "min_chunk_size": min(len(chunk) for chunk in chunks) if chunks else 0,
            "max_chunk_size": max(len(chunk) for chunk in chunks) if chunks else 0,
            "total_characters": len(text),
            "chunks_preview": chunks[:3] if len(chunks) > 3 else chunks
        }
