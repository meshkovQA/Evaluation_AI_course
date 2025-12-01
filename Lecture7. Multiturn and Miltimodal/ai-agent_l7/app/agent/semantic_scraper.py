import os
import re
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langsmith import traceable
from .models import Offer, ExtractionResult
from .http_tools import http_get, strip_boilerplate, absolutize

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=MODEL, temperature=0.0)
parser = PydanticOutputParser(pydantic_object=ExtractionResult)

SYSTEM = (
    "Ты — ИИ-парсер объявлений о жилье. На вход даётся HTML страницы. "
    "Найди карточки объявлений и извлеки для каждой: заголовок, цену (целое число), валюту и ссылку. "
    "Возвращай строго JSON по заданной схеме. Если ссылка относительная — верни как есть."
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human",
     "Базовый URL: {base_url}\n"
     "HTML (обрезано):\n{html}\n"
     "Требуемая схема JSON:\n{schema}\n"
     "Ограничения:\n"
     " - Цена только числом без пробелов.\n"
     " - Валюта как в исходном тексте (₽, RUB, $, USD, €, EUR).\n"
     " - Не более {limit} объявлений.\n"
     "Ответь ТОЛЬКО JSON без комментариев.")
])


def extract_offers_from_html(base_url: str, html: str, limit: int = 50) -> ExtractionResult:
    msgs = PROMPT.format_messages(
        base_url=base_url, html=html, schema=parser.get_format_instructions(), limit=limit
    )
    resp = llm.invoke(msgs)
    text = resp.content
    try:
        data = parser.parse(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        data = parser.parse(m.group(0))
    # absolutize urls
    data.offers = [Offer(title=o.title, price=o.price, currency=o.currency,
                         url=absolutize(base_url, o.url)) for o in data.offers]
    return data


@traceable(name="semantic_scrape")
def extract_offers(url: str, limit: int = 50) -> ExtractionResult:
    raw = http_get(url)
    cleaned = strip_boilerplate(raw)
    return extract_offers_from_html(url, cleaned, limit=limit)
