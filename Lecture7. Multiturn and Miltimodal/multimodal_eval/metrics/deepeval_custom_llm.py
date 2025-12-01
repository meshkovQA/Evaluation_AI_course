"""
Кастомные модели DeepEval для работы через прокси сервер

ProxyLLM - для текстовых метрик (DeepEvalBaseLLM)
ProxyMLLM - для мультимодальных метрик (DeepEvalBaseMLLM)
"""
import os
import base64
from io import BytesIO
from typing import Optional, List, Union, Tuple
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI

from deepeval.models.base_model import DeepEvalBaseLLM, DeepEvalBaseMLLM
from deepeval.test_case import MLLMImage


# Дефолтная конфигурация прокси
PROXY_CONFIG = {
    "api_key": "sk-proxy-your-api-key",
    "base_url": "http://5.11.83.110:8000"
}


class ProxyLLM(DeepEvalBaseLLM):
    """
    Кастомная модель для DeepEval текстовых метрик через прокси
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1
    ):
        self._model_name = model
        self.temperature = temperature

        self.api_key = api_key or os.getenv(
            "PROXY_API_KEY",
            PROXY_CONFIG["api_key"]
        )
        self.base_url = base_url or os.getenv(
            "PROXY_BASE_URL",
            PROXY_CONFIG["base_url"]
        )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.model = self.client

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self._model_name


class ProxyMLLM(DeepEvalBaseMLLM):
    """
    Кастомная МУЛЬТИМОДАЛЬНАЯ модель для DeepEval через прокси.
    Поддерживает изображения + текст.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1
    ):
        self.temperature = temperature

        self.api_key = api_key or os.getenv(
            "PROXY_API_KEY",
            PROXY_CONFIG["api_key"]
        )
        self.base_url = base_url or os.getenv(
            "PROXY_BASE_URL",
            PROXY_CONFIG["base_url"]
        )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        # Вызываем родительский __init__ с model_name
        super().__init__(model_name=model)

    def generate(
        self,
        multimodal_input: List[Union[str, MLLMImage]] = None,
        schema: Optional[BaseModel] = None,
        *,
        # keyword-only для совместимости
        prompt: List[Union[str, MLLMImage]] = None,
    ) -> Union[str, BaseModel]:
        """
        Генерация ответа на мультимодальный запрос.

        Args:
            multimodal_input: Список текстов и/или MLLMImage (positional)
            schema: Pydantic схема для структурированного вывода
            prompt: Альтернативный способ передать input (keyword-only)

        Returns:
            Если schema: объект схемы
            Иначе: строка с ответом
        """
        # Поддержка обоих способов вызова: positional и keyword
        actual_input = multimodal_input if multimodal_input is not None else prompt
        if actual_input is None:
            raise ValueError(
                "Either multimodal_input or prompt must be provided")

        formatted_prompt = self.generate_prompt(actual_input)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=self.temperature
        )

        output = response.choices[0].message.content

        # Если передана schema - пытаемся распарсить JSON
        if schema:
            import json
            try:
                text = output
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]
                json_output = json.loads(text.strip())
                return schema.model_validate(json_output)
            except Exception:
                try:
                    json_output = json.loads(output)
                    return schema.model_validate(json_output)
                except Exception:
                    # Бросаем TypeError чтобы DeepEval использовал fallback
                    raise TypeError(
                        f"Could not parse response as {schema.__name__}")

        # Без schema возвращаем просто строку (не tuple!)
        # DeepEval в fallback режиме ожидает строку для trimAndLoadJson
        return output

    async def a_generate(
        self,
        multimodal_input: List[Union[str, MLLMImage]] = None,
        schema: Optional[BaseModel] = None,
        *,
        prompt: List[Union[str, MLLMImage]] = None,
    ) -> Union[str, BaseModel]:
        """Асинхронная версия generate"""
        return self.generate(multimodal_input, schema, prompt=prompt)

    def generate_prompt(
        self,
        multimodal_input: List[Union[str, MLLMImage]]
    ) -> List[dict]:
        """
        Преобразует список элементов в формат OpenAI Vision API.
        """
        prompt = []
        for ele in multimodal_input:
            if isinstance(ele, str):
                prompt.append({"type": "text", "text": ele})
            elif isinstance(ele, MLLMImage):
                if ele.local:
                    # Локальный файл - кодируем в base64
                    import PIL.Image
                    image = PIL.Image.open(ele.url)
                    visual_dict = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{self._encode_pil_image(image)}"
                        },
                    }
                else:
                    # URL изображения
                    visual_dict = {
                        "type": "image_url",
                        "image_url": {"url": ele.url},
                    }
                prompt.append(visual_dict)
        return prompt

    def _encode_pil_image(self, pil_image) -> str:
        """Кодирует PIL изображение в base64"""
        image_buffer = BytesIO()
        if pil_image.mode in ("RGBA", "LA", "P"):
            pil_image = pil_image.convert("RGB")
        pil_image.save(image_buffer, format="JPEG")
        image_bytes = image_buffer.getvalue()
        return base64.b64encode(image_bytes).decode("utf-8")

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Примерный расчёт стоимости (для gpt-4o-mini)"""
        # Цены примерные, можно уточнить
        pricing = {
            "gpt-4o-mini": {"input": 0.00015 / 1000, "output": 0.0006 / 1000},
            "gpt-4o": {"input": 0.005 / 1000, "output": 0.015 / 1000},
        }
        model_pricing = pricing.get(
            self.model_name,
            pricing["gpt-4o-mini"]
        )
        return input_tokens * model_pricing["input"] + output_tokens * model_pricing["output"]

    def get_model_name(self) -> str:
        return self.model_name


# Фабричные функции
def create_proxy_model(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.1
) -> ProxyMLLM:
    """
    Создаёт ProxyMLLM для мультимодальных метрик DeepEval.
    """
    return ProxyMLLM(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature
    )


def create_proxy_llm(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.1
) -> ProxyLLM:
    """
    Создаёт ProxyLLM для текстовых метрик DeepEval.
    """
    return ProxyLLM(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature
    )
