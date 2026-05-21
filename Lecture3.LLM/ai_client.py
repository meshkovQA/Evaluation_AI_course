# ai_client.py
from openai import OpenAI
from google import genai
from google.genai import types

client = OpenAI(api_key="")

client_google = genai.Client(api_key="")


def google_chat(prompt: str) -> str:
    response = client_google.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=prompt,
    )

    answer = response.text
    print(answer)
    return answer


def google_chat_v2(prompt: str) -> str:
    response = client_google.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1
        )
    )

    answer = response.text
    print(answer)
    return answer


def google_chat_v3(prompt: str, model: str, temperature: float) -> str:
    response = client_google.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature
        )
    )

    answer = response.text
    print(answer)
    return answer


def openai_chat(prompt: str) -> str:
    response = client.responses.create(
        model="gpt-3.5-turbo",
        input=prompt,
    )

    answer = response.output_text
    print(answer)
    return answer


def openai_chat_v2(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    answer = response.choices[0].message.content
    print(answer)
    return answer


def openai_chat_v3(prompt: str, model: str, temperature: 0.1) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )

    answer = response.choices[0].message.content
    return answer

# Примеры использования функций


# translation = openai_chat_v2("Translate to French: Hello world")
# translation = google_chat("Translate to French: Hello world")
# # Результат: "Bonjour le monde"

# # Суммаризация
# summary = openai_chat(
#     "Summarize: [длинный текст о климатических изменениях...]")
# # Результат: краткое изложение

# # Генерация кода
# code = openai_chat("Write a Python function to sort a list")
# # Результат: рабочий код на Python

# # Творческое письмо
# story = openai_chat("Write a short story about a robot")
# Результат: оригинальная история
