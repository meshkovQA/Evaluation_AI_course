"""
Оценка RAG системы с использованием метрик DeepEval
Загружает датасет из Excel, получает ответы от RAG и запускает оценку
"""

import sys
sys.path.insert(
    0, '/Users/aleksandrmeskov/Desktop/AI evaluation/AI_practice/Lecture5/eval_rag')

import time  # noqa: E402
import pandas as pd  # noqa: E402
from typing import List, Union  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402
from deepeval import evaluate  # noqa: E402
from dataset_parser import DatasetParser  # noqa: E402
from rag_connector import RAGConnector  # noqa: E402
from deepeval_custom_llm import create_proxy_model, ProxyLLM  # noqa: E402
from deepeval_evaluator import create_metrics  # noqa: E402


def evaluate_rag_from_excel(
    excel_path: str,
    rag_connector: RAGConnector,
    metrics_list: List[str],
    model: Union[str, ProxyLLM] = "gpt-4o-mini",
    threshold: float = 0.7,
    sleep_time: float = 0.1
):

    # ============= ШАГ 1: Парсим Excel =============
    print(f" Шаг 1: Парсинг Excel файла...")

    parser = DatasetParser()
    df = parser.load_dataset(excel_path)

    if df is None:
        print("❌ Ошибка загрузки датасета")
        return

    # Превью
    parser.preview_dataset(df, n=2)

    # ============= ШАГ 2: Получаем вопросы и expected ответы =============
    print(f"Шаг 2: Извлечение вопросов из датасета...")

    questions = parser.get_questions(df)
    expected_responses = parser.get_expected_responses(df)

    print(f"Извлечено {len(questions)} вопросов")

    # ============= ШАГ 3: Задаем вопросы в RAG =============
    print(f"Шаг 3: Получение ответов от RAG системы...")

    test_cases = []

    for i, question in enumerate(questions, 1):
        print(f"\n  [{i}/{len(questions)}] {question[:60]}...")

        # Запрос к RAG
        rag_response = rag_connector.query(question)

        # Проверяем на ошибки
        if 'error' in rag_response:
            print(f"Ошибка RAG: {rag_response['error']}")
            continue

        # Извлекаем данные
        actual_output = rag_response.get('content', '')
        sources = rag_response.get('sources', [])
        retrieval_context = [s.get('content', '')
                             for s in sources if s.get('content')]

        print(f"Ответ: {actual_output[:60]}...")
        print(f"Контекстов: {len(retrieval_context)}")

        # Получаем expected_output
        expected = expected_responses[i-1] if i - \
            1 < len(expected_responses) else ""

        # Создаем LLMTestCase для DeepEval
        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            expected_output=expected,
            retrieval_context=retrieval_context
        )

        test_cases.append(test_case)

        # Пауза между запросами
        time.sleep(sleep_time)

    print(f"\n✅ Создано {len(test_cases)} тест кейсов")

    # ============= ШАГ 4: Создаем метрики =============
    print(f"\n📊 Шаг 4: Создание метрик для оценки...")
    print(f"   Метрики: {metrics_list}")
    print(f"   Порог: {threshold}")

    metrics = create_metrics(
        model=model,
        metrics_list=metrics_list,
        threshold=threshold
    )

    print(f"✅ Создано {len(metrics)} метрик")

    # ============= ШАГ 5: Запускаем оценку =============
    print(f"\n🧪 Шаг 5: Запуск оценки...")

    try:
        # DeepEval автоматически выведет результаты
        evaluate(
            test_cases=test_cases,
            metrics=metrics
        )

        print(f"\n✅ Оценка завершена!")
        print(f"🌐 Результаты сохранены на https://app.confident-ai.com")

    except Exception as e:
        print(f"❌ Ошибка при оценке: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":

    # ============= НАСТРОЙКА =============

    # 1. Инициализируем RAG коннектор
    rag_connector = RAGConnector(
        endpoint_url="http://5.11.83.110:8002/api/v1/chat/",
        api_key="rag-api-key",
        timeout=30
    )

    # 2. Создаем прокси модель для метрик (или используйте "gpt-4o-mini")
    proxy_model = create_proxy_model(
        model="gpt-4o-mini",
        api_key="sk-proxy-your-key",
        base_url="http://5.11.83.110:8000"
    )

    # 3. Указываем путь к Excel файлу
    excel_path = "data/evaluation_dataset.xlsx"

    # 4. Выбираем метрики для оценки
    metrics_to_use = [
        'answer_relevancy',
        'faithfulness',
        'contextual_relevancy'
    ]

    # ============= ЗАПУСК ОЦЕНКИ =============

    evaluate_rag_from_excel(
        excel_path=excel_path,
        rag_connector=rag_connector,
        metrics_list=metrics_to_use,
        model=proxy_model,  # Или "gpt-4o-mini" для обычной OpenAI
        threshold=0.7,
        sleep_time=0.1
    )
