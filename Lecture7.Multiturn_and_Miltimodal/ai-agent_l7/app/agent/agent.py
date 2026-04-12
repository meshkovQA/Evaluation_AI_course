import os
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langfuse import observe, get_client, propagate_attributes
from langfuse.langchain import CallbackHandler
from .semantic_scraper import extract_offers
from .tools_ext import compute_stats, filter_offers, normalize_offers_currency
from contextvars import ContextVar
from threading import Lock

# Импорт для веб-поиска через OpenAI Responses API
from openai import OpenAI

_openai_client = OpenAI()

# Хранилище состояний по session_id
_SESSION_STATES: Dict[str, Dict[str, Any]] = {}
_SESSION_LOCK = Lock()

# Контекстная переменная для передачи session_id в tools
_current_session_id: ContextVar[Optional[str]] = ContextVar('current_session_id', default=None)


def _get_session_state(session_id: Optional[str] = None) -> Dict[str, Any]:
    """Получить состояние для текущей сессии"""
    sid = session_id or _current_session_id.get()
    if not sid:
        sid = "__default__"

    with _SESSION_LOCK:
        if sid not in _SESSION_STATES:
            _SESSION_STATES[sid] = {"offers": []}
        return _SESSION_STATES[sid]


def _set_session_offers(offers: List[Dict[str, Any]], session_id: Optional[str] = None):
    """Сохранить offers для текущей сессии"""
    state = _get_session_state(session_id)
    state["offers"] = offers

SYSTEM = (
    """Ты — многоцелевой агент мониторинга цен на жильё.

ВАЖНО: У тебя есть память о ранее извлечённых объявлениях в рамках сессии.
Если пользователь задаёт уточняющий вопрос (например "какие из них...", "покажи ещё...", "отфильтруй...")
и НЕ дал новый URL — используй ранее извлечённые данные, вызывая filter_offers или compute_stats БЕЗ параметра offers.

Правила работы:
1) Если пользователь дал URL — извлеки объявления через extract_offers
2) Если пользователь спрашивает про "эти квартиры", "из них", "ранее найденные" —
   просто вызови filter_offers/compute_stats без offers, они возьмут данные из памяти
3) Если просили фильтровать — применяй filter_offers с нужными параметрами
4) Если просили в другой валюте — normalize_offers_currency
5) Если нужна статистика — compute_stats
6) Если пользователь спрашивает о расположении квартир, районах, инфраструктуре,
   транспорте, близости к центру, безопасности района или другой информации,
   которой нет в объявлениях — используй web_search для поиска в интернете.

ВАЖНО для web_search:
- ОБЯЗАТЕЛЬНО включай в поисковый запрос КОНТЕКСТ из текущего разговора!
- Если пользователь спрашивает о найденных квартирах — включи в запрос:
  * Город/страну из URL или объявлений (например "New York", "Москва")
  * Конкретные адреса/улицы из найденных квартир
  * Суть вопроса пользователя
- Примеры:
  * Квартиры в NYC, вопрос "какие ближе к центру?" → запрос: "East 53rd street New York Manhattan расстояние до центра"
  * Квартиры в Москве, вопрос "где лучше инфраструктура?" → запрос: "улица Тверская Москва инфраструктура метро магазины"
- НИКОГДА не делай запросы без контекста! Запрос "квартиры ближе к центру" — ПЛОХО, нет привязки к конкретным адресам

Всегда вызывай инструменты именованными аргументами строго по их JSON-схеме.
Отвечай кратко и на том же языке, на котором написан вопрос пользователя. При необходимости показывай 3–5 примеров ссылок."""
)


@observe(name="extract_offers")
def _extract_offers_tool(url: str, limit: int = 50) -> list[dict]:
    res = extract_offers(url, limit=limit)
    out = [o.dict() for o in res.offers]
    # Сохраняем в состояние текущей сессии
    _set_session_offers(out)
    return out


@observe(name="filter_offers")
def _filter_offers_tool(offers: Optional[List[Dict[str, Any]]] = None, min_price: Optional[int] = None,
                        max_price: Optional[int] = None,
                        text_contains: str = "") -> List[dict]:
    if offers is None:
        # Берём из состояния текущей сессии
        offers = _get_session_state().get("offers", [])
    return filter_offers(offers, min_price, max_price, text_contains)


@observe(name="normalize_currency")
def _normalize_offers_currency_tool(offers: Optional[List[Dict[str, Any]]] = None,
                                    target_currency: str = "RUB") -> List[dict]:
    if offers is None:
        # Берём из состояния текущей сессии
        offers = _get_session_state().get("offers", [])
    return normalize_offers_currency(offers, target_currency)


@observe(name="compute_stats")
def _compute_stats_tool(prices: Optional[List[int]] = None,
                        offers: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
    if prices is None and offers is None:
        # Берём из состояния текущей сессии
        offers = _get_session_state().get("offers", [])
    if prices is None and offers is not None:
        prices = [int(o.get("price", 0)) for o in offers]
    return compute_stats(prices or [])


@observe(name="web_search")
def _web_search_tool(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Поиск информации в интернете через OpenAI Responses API с web_search.
    Используется для получения дополнительной информации о районах,
    инфраструктуре, транспорте и т.д.
    """
    try:
        response = _openai_client.responses.create(
            model="gpt-4o-mini",
            tools=[{"type": "web_search_preview"}],
            input=query
        )

        # Извлекаем текстовый ответ и источники
        result = {
            "answer": "",
            "sources": []
        }

        for item in response.output:
            # Текстовый ответ
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        result["answer"] = content.text
                        # Извлекаем аннотации (источники)
                        if hasattr(content, 'annotations') and content.annotations:
                            for ann in content.annotations:
                                if hasattr(ann, 'url'):
                                    result["sources"].append({
                                        "title": getattr(ann, 'title', ''),
                                        "url": ann.url
                                    })

        return result if result["answer"] else {"info": "No results found"}
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}


class ExtractOffersArgs(BaseModel):
    url: str = Field(..., description="Страница с объявлениями")
    limit: int = Field(50, description="Максимум объявлений")


class FilterOffersArgs(BaseModel):
    offers: Optional[List[Dict[str, Any]]] = Field(
        None, description="Список объявлений {title, price, currency, url}. Если не задан — берём последние извлечённые.")
    min_price: Optional[int] = Field(None, description="Минимальная цена")
    max_price: Optional[int] = Field(None, description="Максимальная цена")
    text_contains: Optional[str] = Field(
        "", description="Подстрока в заголовке, например 'однокомнатная'")


class NormalizeCurrencyArgs(BaseModel):
    offers: Optional[List[Dict[str, Any]]] = Field(
        None, description="Список объявлений. Если не задан — берём последние извлечённые.")
    target_currency: str = Field(
        "RUB", description="Целевая валюта: RUB|USD|EUR")


class ComputeStatsArgs(BaseModel):
    prices: Optional[List[int]] = Field(None, description="Список цен")
    offers: Optional[List[Dict[str, Any]]] = Field(
        None, description="Можно передать объявления вместо списка цен")


class WebSearchArgs(BaseModel):
    query: str = Field(..., description="Поисковый запрос на русском или английском языке")
    max_results: int = Field(5, description="Максимум результатов (1-10)")


tools = [
    StructuredTool.from_function(
        func=_extract_offers_tool,
        name="extract_offers",
        args_schema=ExtractOffersArgs,
        description="Извлечь объявления (title, price, currency, url) с указанной страницы."
    ),
    StructuredTool.from_function(
        func=_compute_stats_tool,
        name="compute_stats",
        args_schema=ComputeStatsArgs,
        description="Подсчитать статистику по списку цен или по объявлениям: {min,max,avg,median}."
    ),
    StructuredTool.from_function(
        func=_filter_offers_tool,
        name="filter_offers",
        args_schema=FilterOffersArgs,
        description="Отфильтровать объявления по min_price/max_price и/или по подстроке в заголовке."
    ),
    StructuredTool.from_function(
        func=_normalize_offers_currency_tool,
        name="normalize_offers_currency",
        args_schema=NormalizeCurrencyArgs,
        description="Сконвертировать цены объявлений в целевую валюту (RUB|USD|EUR)."
    ),
    StructuredTool.from_function(
        func=_web_search_tool,
        name="web_search",
        args_schema=WebSearchArgs,
        description="Поиск информации в интернете. Используй для вопросов о районах, "
                    "инфраструктуре, транспорте, расположении, безопасности района, "
                    "близости к центру города и другой информации, которой нет в объявлениях."
    ),
]

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human",
     "Задача пользователя: {question}\n"
     "Новые ссылки (может быть пусто): {urls}\n\n"
     "КОНТЕКСТ СЕССИИ:\n{session_context}\n\n"
     "ПОМНИ: Если ссылок нет, но пользователь спрашивает про ранее найденные квартиры — "
     "используй filter_offers или compute_stats БЕЗ параметра offers, данные уже в памяти.\n"
     "При использовании web_search — ОБЯЗАТЕЛЬНО учитывай контекст сессии!"),
    MessagesPlaceholder("agent_scratchpad"),
])

llm = ChatOpenAI(model=os.getenv(
    "OPENAI_MODEL", "gpt-4o-mini"), temperature=0.0)
agent = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    verbose=False,
    return_intermediate_steps=True,
)


def _build_session_context(session_id: Optional[str]) -> str:
    """Формирует строку контекста сессии для передачи в промпт"""
    state = _get_session_state(session_id)
    offers = state.get("offers", [])

    if not offers:
        return "Пока нет данных"

    # Формируем краткую сводку по найденным квартирам
    context_parts = []
    context_parts.append(f"Найдено {len(offers)} объявлений")

    # Извлекаем адреса/заголовки (первые 5 для примера)
    titles = [o.get("title", "") for o in offers[:5]]
    if titles:
        context_parts.append(f"Примеры адресов: {', '.join(titles)}")

    # Извлекаем город из URL если есть
    urls_in_offers = [o.get("url", "") for o in offers[:1]]
    if urls_in_offers and urls_in_offers[0]:
        context_parts.append(f"Источник: {urls_in_offers[0]}")

    return "\n".join(context_parts)


@observe(name="run_question")
def run_question(question: str, urls: list[str], max_items: int = 50,
                 session_id: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
    langfuse_client = get_client()

    handler = CallbackHandler()

    config = {
        "callbacks": [handler],
        "metadata": {},
        "tags": []
    }

    # Добавляем session_id и user_id в metadata для LangSmith
    if session_id:
        config["metadata"]["session_id"] = session_id
        config["tags"].append(f"session:{session_id}")
    if user_id:
        config["metadata"]["user_id"] = user_id
        config["tags"].append(f"user:{user_id}")

    # Устанавливаем session_id в контекстную переменную для tools
    token = _current_session_id.set(session_id)

    # Формируем контекст сессии
    session_context = _build_session_context(session_id)

    try:
        # Начинаем трассу / наблюдение
        with langfuse_client.start_as_current_observation(
            as_type="span", name="agent_execution"
        ):
            # Пропагируем session_id и user_id (если они заданы)
            attrs: Dict[str, str] = {}
            if session_id:
                attrs["session_id"] = session_id
            if user_id:
                attrs["user_id"] = user_id

            if attrs:
                with propagate_attributes(**attrs):
                    result = executor.invoke(
                        {"question": question, "urls": urls, "max_items": max_items, "session_context": session_context},
                        config=config
                    )
            else:
                result = executor.invoke(
                    {"question": question, "urls": urls, "max_items": max_items, "session_context": session_context},
                    config=config
                )
    finally:
        # Сбрасываем контекстную переменную
        _current_session_id.reset(token)

    tools_used: List[str] = []
    intermediate_steps = result.get("intermediate_steps") or []
    for step in intermediate_steps:
        # Шаг имеет формат (AgentAction, tool_output)
        try:
            action = step[0]
            tool_name = getattr(action, "tool", None)
            if tool_name:
                tools_used.append(tool_name)
        except Exception:
            continue

    return {
        "output": result.get("output"),
        "tools_used": tools_used,
    }
