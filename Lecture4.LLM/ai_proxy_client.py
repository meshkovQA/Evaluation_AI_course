# ai_proxy_client.py
from openai import OpenAI
import os
import httpx


# OpenAI через прокси
client = OpenAI(
    # Личный ключ от прокси
    api_key=os.getenv(
        "PROXY_API_KEY", "sk-proxy-"),
    # прокси сервер
    base_url=os.getenv("PROXY_BASE_URL", "http://5.11.83.110:8000")
)

# ============= Google через прокси =============
GOOGLE_PROXY_API_KEY = os.getenv(
    "GOOGLE_PROXY_API_KEY", "sk-google-proxy-")
GOOGLE_PROXY_BASE_URL = os.getenv(
    "GOOGLE_PROXY_BASE_URL", "http://5.11.83.110:8001")


def openai_chat(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.choices[0].message.content
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


def openai_chat_v3(prompt: str, model: str, temperature: float = 0.1) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )

    answer = response.choices[0].message.content
    return answer


def google_chat(prompt: str) -> str:
    url = f"{GOOGLE_PROXY_BASE_URL}/models/gemini-2.0-flash-lite:generateContent"

    headers = {
        "Authorization": f"Bearer {GOOGLE_PROXY_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }]
    }

    with httpx.Client(timeout=300.0) as client_http:
        response = client_http.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()

        answer = result["candidates"][0]["content"]["parts"][0]["text"]
        print(answer)
        return answer


def google_chat_v2(prompt: str) -> str:
    url = f"{GOOGLE_PROXY_BASE_URL}/models/gemini-2.0-flash-lite:generateContent"

    headers = {
        "Authorization": f"Bearer {GOOGLE_PROXY_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.1
        }
    }

    with httpx.Client(timeout=300.0) as client_http:
        response = client_http.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()

        answer = result["candidates"][0]["content"]["parts"][0]["text"]
        print(answer)
        return answer


def google_chat_v3(prompt: str, model: str, temperature: float) -> str:
    url = f"{GOOGLE_PROXY_BASE_URL}/models/{model}:generateContent"

    headers = {
        "Authorization": f"Bearer {GOOGLE_PROXY_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": temperature
        }
    }

    with httpx.Client(timeout=300.0) as client_http:
        response = client_http.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()

        answer = result["candidates"][0]["content"]["parts"][0]["text"]
        print(answer)
        return answer
