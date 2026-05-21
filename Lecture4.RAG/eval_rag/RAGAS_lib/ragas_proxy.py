"""
RAGAS с поддержкой прокси через LangChain
"""
from typing import List, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


def create_proxy_llm(
    model: str = "gpt-4o-mini",
    api_key: str = "sk-proxy-your-key",
    base_url: str = "http://5.11.83.110:8000",
    temperature: float = 0.0
):
    """
    Создать LLM для RAGAS через прокси

    Args:
        model: Название модели
        api_key: API ключ прокси
        base_url: URL прокси сервера
        temperature: Температура

    Returns:
        LangchainLLMWrapper для RAGAS
    """
    # LangChain OpenAI с кастомным base_url
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,  # ← Ваш прокси!
        temperature=temperature
    )

    # Оборачиваем для RAGAS
    return LangchainLLMWrapper(llm)


def create_proxy_embeddings(
    api_key: str = "sk-proxy-your-key",
    base_url: str = "http://5.11.83.110:8000"
):
    """
    Создать Embeddings для RAGAS через прокси

    Args:
        api_key: API ключ прокси
        base_url: URL прокси сервера

    Returns:
        LangchainEmbeddingsWrapper для RAGAS
    """
    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        base_url=base_url  # ← Ваш прокси!
    )

    return LangchainEmbeddingsWrapper(embeddings)


# # ==================== ПРИМЕР ИСПОЛЬЗОВАНИЯ ====================

# if __name__ == "__main__":
#     from ragas.dataset_schema import SingleTurnSample
#     from ragas.metrics import Faithfulness
#     import asyncio

#     print("🔄 Тестирование RAGAS с прокси...")

#     # Создаем прокси LLM
#     proxy_llm = create_proxy_llm(
#         model="gpt-4o-mini",
#         api_key="sk-proxy-your-key",
#         base_url="http://5.11.83.110:8000"
#     )

#     print(f"✅ Прокси LLM создан")

#     # Создаем метрику с прокси
#     metric = Faithfulness(llm=proxy_llm)

#     # Тестовые данные
#     sample = SingleTurnSample(
#         user_input="What is the capital of France?",
#         response="Paris is the capital of France.",
#         retrieved_contexts=["Paris is the capital and largest city of France."]
#     )

#     # Оцениваем
#     async def test():
#         score = await metric.single_turn_ascore(sample)
#         print(f"\n✅ Faithfulness score: {score:.3f}")

#     asyncio.run(test())
