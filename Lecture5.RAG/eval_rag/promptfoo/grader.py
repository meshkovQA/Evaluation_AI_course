# grader.py
import sys
from openai import OpenAI

# Ваш рабочий клиент через прокси
client = OpenAI(
    api_key="sk-proxy-....",
    base_url="http://5.11.83.110:8000"
)


def call_api(prompt, options=None, context=None):
    """
    Оценщик качества ответа LLM
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        return {
            "output": response.choices[0].message.content
        }

    except Exception as e:
        print(f"❌ Error in call_api: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {
            "output": f"Error: {str(e)}"
        }


def call_embedding_api(prompt, options=None, context=None):
    """
    Оценщик качества ответа LLM - получение эмбеддингов
    """
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=prompt if isinstance(prompt, str) else prompt[0]
        )

        # Возвращаем embedding vector
        return {
            "embedding": response.data[0].embedding
        }

    except Exception as e:
        print(f"❌ Error in call_embedding_api: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {
            "embedding": []
        }
