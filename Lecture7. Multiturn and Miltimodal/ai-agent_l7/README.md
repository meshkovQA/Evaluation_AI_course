# 🏠 Semantic Price Scraper Agent

AI-агент для мониторинга цен на жильё.  
Построен на **LangChain** + **OpenAI LLM**, интегрирован с **LangSmith** и **Langfuse** для полного трейсинга и аналитики.

---

## 📖 Возможности

- 🔎 Извлечение объявлений с сайтов (title, price, currency, url)
- 🧹 Фильтрация объявлений по цене и ключевым словам
- 💱 Конвертация цен в разные валюты (USD, RUB, EUR)
- 📊 Подсчёт статистики по ценам (min, max, avg, median)
- 🌐 **Веб-поиск** - получение информации о районах, инфраструктуре, транспорте
- 💾 Запоминание последнего результата для повторного анализа
- 📈 **Полный трейсинг с Langfuse** - отслеживание всех вызовов LLM и инструментов
- 🔄 **Управление сессиями** - группировка связанных запросов
- 👥 **Поддержка user_id** - аналитика по пользователям

---

## 🛠 Используемые инструменты (Tools)

Агент использует 5 встроенных инструментов, каждый из которых обёрнут в Langfuse `@observe` для полного трейсинга:

### 1. `extract_offers`
Извлекает объявления с указанной страницы.  
**Вход:** URL, limit  
**Выход:** массив объявлений `{title, price, currency, url}`

### 2. `filter_offers`
Фильтрует объявления по цене и ключевым словам.  
**Вход:** список объявлений, min/max цена, текстовый фильтр  
**Выход:** отфильтрованный список

### 3. `normalize_offers_currency`
Конвертирует цены объявлений в выбранную валюту (RUB / USD / EUR).  
**Вход:** объявления + целевая валюта  
**Выход:** обновлённые объявления

### 4. `compute_stats`
Строит статистику по ценам.
**Вход:** либо список цен, либо список объявлений
**Выход:** `{min, max, avg, median}`

### 5. `web_search`
Поиск информации в интернете через OpenAI Responses API.
Используется для получения дополнительной информации о районах, инфраструктуре, транспорте, безопасности и других данных, которых нет в объявлениях.
**Вход:** query (поисковый запрос), max_results (1-10)
**Выход:** `{answer, sources[], raw_annotations[]}`

Агент автоматически учитывает контекст сессии при формировании поисковых запросов (город, адреса из найденных квартир).

Агент автоматически сохраняет последний список объявлений в сессию и подставляет его в инструменты, если пользователь не передаёт `offers` вручную.

---

## 🔄 Как работает агент (процесс обработки)

1. **Получение вопроса**  
   Пользователь отправляет запрос с вопросом и (опционально) ссылками.

2. **Создание trace в Langfuse и Langsmith**  
   Внутри `run_question()` запускается наблюдение `agent_execution`, в которое автоматически попадают:
   - вызовы LLM
   - вызовы инструментов
   - промежуточные шаги агента (Chain-of-Thought скрыт)
   - session_id и user_id (если переданы)

3. **Построение плана действий**  
   LLM использует системный prompt и выбирает, какие инструменты вызывать.

4. **Вызов инструментов**
   В зависимости от вопроса:
   - извлечение объявлений
   - фильтрация
   - конвертация валют
   - статистика
   - веб-поиск (для информации о районах, инфраструктуре и т.д.)
   Все шаги записываются в `intermediate_steps`.

5. **Формирование ответа**  
   Агент собирает итоговый результат + массив `tools_used` в порядке вызова.

6. **Возврат результата клиенту**  
   `/ask` возвращает:
   ```json
   {
     "output": "ответ агента",
     "tools_used": ["extract_offers", "filter_offers"],
     "_session_id": "...",
     "_user_id": "..."
   }
   ```

Эта информация может использоваться для тестирования, аналитики и верификации поведения агента.

---


## 📡 API эндпоинты

### 1. `/ask` — задать вопрос агенту с трейсингом

**POST**
```json
{
  "question": "Найди квартиры дешевле 2000 долларов",
  "urls": ["https://example.com"],
  "session_id": "uuid-сессии",  // опционально
  "user_id": "user123"          // опционально
}
```

**С использованием HTTP хедеров (рекомендуется):**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: 123e4567-e89b-12d3-a456-426614174000" \
  -H "X-User-Id: user123" \
  -d '{
    "question": "Найди квартиры дешевле 2000 долларов",
    "urls": ["https://www.rentalads.com/apartments-for-rent/ny/new-york/"]
  }'
```

### 2. `/session/new` — создать новую сессию

**GET**
```bash
curl http://localhost:8000/session/new
```

Ответ:
```json
{
  "session_id": "123e4567-e89b-12d3-a456-426614174000",
  "message": "Используйте этот ID в хедере X-Session-Id для группировки запросов"
}
```

### 3. `/monitor` — прямое извлечение с сайта

**POST**
```bash
curl -X POST http://localhost:8000/monitor \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: your-session-id" \
  -d '{
    "url": "https://www.rentalads.com/apartments-for-rent/ny/new-york/",
    "max_items": 50
  }'
```

---

## 🎯 Работа с сессиями

### Пример полного workflow с одной сессией:
```python
import requests
import json

# 1. Создаем новую сессию
session_resp = requests.get("http://localhost:8000/session/new")
session_id = session_resp.json()["session_id"]
print(f"Сессия создана: {session_id}")

headers = {
    "Content-Type": "application/json",
    "X-Session-Id": session_id,
    "X-User-Id": "user123"
}

# 2. Извлекаем объявления
resp1 = requests.post(
    "http://localhost:8000/ask",
    headers=headers,
    json={
        "question": "Извлеки квартиры с сайта",
        "urls": ["https://www.rentalads.com/apartments-for-rent/ny/new-york/"]
    }
)

# 3. Фильтруем по цене (в той же сессии)
resp2 = requests.post(
    "http://localhost:8000/ask",
    headers=headers,
    json={
        "question": "Покажи только квартиры дешевле 2000 долларов"
    }
)

# 4. Конвертируем в рубли (в той же сессии)
resp3 = requests.post(
    "http://localhost:8000/ask",
    headers=headers,
    json={
        "question": "Переведи цены в рубли"
    }
)

# 5. Получаем статистику (в той же сессии)
resp4 = requests.post(
    "http://localhost:8000/ask",
    headers=headers,
    json={
        "question": "Покажи статистику по ценам"
    }
)

# 6. Узнаём о районе (web_search в контексте сессии)
resp5 = requests.post(
    "http://localhost:8000/ask",
    headers=headers,
    json={
        "question": "Какой район лучше для семьи с детьми?"
    }
)
# Агент автоматически добавит в поисковый запрос контекст (New York, найденные адреса)
```

---

## 🐳 Docker Compose варианты

### Cloud Langfuse (рекомендуется):
```
bash
docker compose up -d --build
```

---

## ⚠️ Ограничения

- Агент работает только с открытыми страницами (без авторизации)
- HTML структура сайтов может меняться, что сломает парсинг
- Для работы требуется действующий OPENAI_API_KEY
- Langfuse трейсинг добавляет небольшую задержку (~50-100ms)

## 📚 Полезные ссылки

- [Langfuse Documentation](https://docs.langfuse.com)
- [LangChain Integration Guide](https://docs.langfuse.com/integrations/langchain)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com)