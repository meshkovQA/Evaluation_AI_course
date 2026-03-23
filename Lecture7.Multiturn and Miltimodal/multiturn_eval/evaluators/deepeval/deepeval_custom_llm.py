"""
Кастомная модель DeepEval для работы через прокси сервер
Следует официальной документации DeepEval
"""
import os
from typing import Optional
from openai import OpenAI
from deepeval.models.base_model import DeepEvalBaseLLM


class ProxyLLM(DeepEvalBaseLLM):
    """
    Кастомная модель для DeepEval, которая работает через прокси сервер
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1
    ):
        # Сохраняем название модели
        self._model_name = model
        self.temperature = temperature

        # Получаем настройки из переменных окружения или используем переданные
        self.api_key = api_key or os.getenv(
            "PROXY_API_KEY",
            "sk-proxy-your-key"
        )
        self.base_url = base_url or os.getenv(
            "PROXY_BASE_URL",
            "http://5.11.83.110:8000"
        )

        # Создаем клиент OpenAI с прокси настройками
        # ВАЖНО: Ваш прокси использует Bearer токен в Authorization
        self.client = OpenAI(
            # Это попадет в Authorization: Bearer {api_key}
            api_key=self.api_key,
            base_url=self.base_url
        )

        # НЕ вызываем super().__init__() - вместо этого устанавливаем self.model вручную
        self.model = self.client

    def load_model(self):
        """
        Загрузка модели (требуется DeepEval)
        Возвращает клиент OpenAI
        """
        return self.client

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"❌ Ошибка при генерации через прокси: {e}")
            raise

    async def a_generate(self, prompt: str) -> str:
        # DeepEval может использовать async для параллельной оценки
        # Для простоты используем синхронную версию
        return self.generate(prompt)

    def get_model_name(self) -> str:
        """
        Получить название модели (требуется DeepEval)
        """
        return self._model_name


def create_proxy_model(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.1
) -> ProxyLLM:

    return ProxyLLM(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature
    )
