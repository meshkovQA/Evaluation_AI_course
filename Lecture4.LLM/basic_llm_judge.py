# from ai_proxy_client import google_chat_v3
from ai_client import google_chat_v3

"""
В случае использования proxy, мы полностью удаляем импорт обычного ai_client или комментируем его.
"""


def basic_llm_judge(question, answer):
    """
    Простая оценка ответа с помощью LLM-судьи.
    """
    prompt = f"""
Оцени качество ответа на вопрос от 1 до 10:

Вопрос: {question}
Ответ: {answer}

Критерии: точность, полнота, читаемость
Верни только число от 1 до 10 и краткое объяснение.
"""

    response = google_chat_v3(
        prompt, model="gemini-2.0-flash", temperature=0.1)
    return response


# Пример использования
question = "Как работает фотосинтез?"
answer = "Фотосинтез -- это процесс, при котором растения используют свет для создания глюкозы из CO2 и воды."

result = basic_llm_judge(question, answer)
print(f"Базовая оценка: {result}")
