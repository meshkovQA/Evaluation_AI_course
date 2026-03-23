"""
End-to-End RAG Evaluation Pipeline with eval_lib
Загружает датасет из Excel, получает ответы от RAG и запускает оценку
"""

import sys
sys.path.insert(
    0, '/Users/aleksandrmeskov/Desktop/AI evaluation/AI_practice/Lecture5/eval_rag')

sys.path.insert(0, '/Users/aleksandrmeskov/Desktop/Projects/Eval-ai-library')

import asyncio  # noqa: E402
import time  # noqa: E402
import pandas as pd  # noqa: E402
from typing import List  # noqa: E402
from eval_lib import (  # noqa: E402
    evaluate,
    EvalTestCase,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    CustomLLMClient
)
from dataset_parser import DatasetParser  # noqa: E402
from rag_connector import RAGConnector  # noqa: E402
from custom_proxy_llm import create_proxy_llm  # noqa: E402


async def evaluate_rag_from_excel(
    excel_path: str,
    rag_connector: RAGConnector,
    metrics_list: List[str],
    model: str | CustomLLMClient = "gpt-4o-mini",
    threshold: float = 0.7,
    temperature: float = 0.5,
    sleep_time: float = 0.1,
    verbose: bool = True,
    show_dashboard: bool = False,
    session_name: str = "Evaluation Session"
):

    # ============= ШАГ 1: Парсим Excel =============
    print(f"\n Шаг 1: Парсинг Excel файла...")

    parser = DatasetParser()
    df = parser.load_dataset(excel_path)

    if df is None:
        print("❌ Ошибка загрузки датасета")
        return

    # Показываем информацию
    info = parser.validate_dataset(df)
    print(f"\n Информация о датасете:")
    print(f"   Всего строк: {info['total_rows']}")
    print(f"   Валидных пар: {info['valid_pairs']}")

    # Превью
    parser.preview_dataset(df, n=2)

    # ============= ШАГ 2: Получаем вопросы и expected ответы =============
    print(f"\n Шаг 2: Извлечение вопросов из датасета...")

    questions = parser.get_questions(df)
    expected_responses = parser.get_expected_responses(df)

    print(f" Извлечено {len(questions)} вопросов")

    # ============= ШАГ 3: Задаем вопросы в RAG =============
    print(f"\n Шаг 3: Получение ответов от RAG системы...")

    test_cases = []

    for i, question in enumerate(questions, 1):
        print(f"\n  [{i}/{len(questions)}] {question[:60]}...")

        # Запрос к RAG
        rag_response = rag_connector.query(question)

        # Проверяем на ошибки
        if 'error' in rag_response:
            print(f"      ⚠️  Ошибка RAG: {rag_response['error']}")
            continue

        # Извлекаем данные
        actual_output = rag_response.get('content', '')
        sources = rag_response.get('sources', [])
        retrieval_context = [s.get('content', '')
                             for s in sources if s.get('content')]

        print(f"      ✅ Ответ: {actual_output[:60]}...")
        print(f"      📚 Контекстов: {len(retrieval_context)}")

        # Получаем expected_output
        expected = expected_responses[i-1] if i - \
            1 < len(expected_responses) else ""

        # Создаем EvalTestCase
        test_case = EvalTestCase(
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
    print(f"\n Шаг 4: Создание метрик для оценки...")
    print(f"   Метрики: {metrics_list}")
    print(f"   Порог: {threshold}")
    print(f"   Температура: {temperature}")

    # Определяем тип модели для вывода
    model_name = model.get_model_name() if isinstance(
        model, CustomLLMClient) else model
    print(f"   Модель: {model_name}")

    metrics = []
    metric_classes = {
        'answer_relevancy': AnswerRelevancyMetric,
        'faithfulness': FaithfulnessMetric,
        'contextual_relevancy': ContextualRelevancyMetric
    }

    for metric_name in metrics_list:
        if metric_name in metric_classes:
            metric = metric_classes[metric_name](
                model=model,
                threshold=threshold,
                temperature=temperature,
                verbose=verbose
            )
            metrics.append(metric)
            print(f"   ✅ Добавлена метрика: {metric_name}")

    print(f"\n✅ Создано {len(metrics)} метрик")

    # ============= ШАГ 5: Запускаем оценку =============
    print(f"\n🧪 Шаг 5: Запуск оценки...")
    print(f"   Тест кейсов: {len(test_cases)}")
    print(f"   Метрик: {len(metrics)}")

    try:
        # Запускаем оценку
        results = await evaluate(
            test_cases=test_cases,
            metrics=metrics,
            verbose=verbose,
            show_dashboard=show_dashboard,
            session_name=session_name
        )

        return results

    except Exception as e:
        print(f"❌ Ошибка при оценке: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==================== ТЕСТОВЫЕ СЦЕНАРИИ ====================

async def scenario_1_standard_openai():
    """Сценарий 1: Стандартная OpenAI модель"""

    print("\n" + "="*70)
    print("📋 СЦЕНАРИЙ 1: Оценка с стандартной OpenAI моделью")
    print("="*70)

    rag_connector = RAGConnector(
        endpoint_url="http://5.11.83.110:8002/api/v1/chat/",
        api_key="rag-api-key",
        timeout=30
    )

    excel_path = "data/evaluation_dataset.xlsx"

    metrics_to_use = [
        'answer_relevancy',
        'faithfulness',
        'contextual_relevancy'
    ]

    results = await evaluate_rag_from_excel(
        excel_path=excel_path,
        rag_connector=rag_connector,
        metrics_list=metrics_to_use,
        model="gpt-4o-mini",  # Стандартная OpenAI
        threshold=0.7,
        temperature=0.5,
        sleep_time=0.1,
        verbose=True,
        show_dashboard=True,
        session_name="OpenAI Model Evaluation"

    )

    return results


async def scenario_2_custom_model():
    """Сценарий 2: Кастомная модель (прокси)"""

    # Создаем кастомную модель
    custom_model = create_proxy_llm(
        model="gpt-4o-mini",
        api_key="sk-proxy-your-key",
        base_url="http://5.11.83.110:8000"
    )

    rag_connector = RAGConnector(
        endpoint_url="http://5.11.83.110:8002/api/v1/chat/",
        api_key="rag-api-key",
        timeout=30
    )

    excel_path = "data/evaluation_dataset.xlsx"

    metrics_to_use = [
        'answer_relevancy',
        'faithfulness',
        'contextual_relevancy'
    ]

    results = await evaluate_rag_from_excel(
        excel_path=excel_path,
        rag_connector=rag_connector,
        metrics_list=metrics_to_use,
        model=custom_model,  # Кастомная модель
        threshold=0.7,
        temperature=0.5,
        sleep_time=0.1,
        verbose=True,
        show_dashboard=True,
        session_name="OpenAI Model Evaluation"
    )

    return results


async def scenario_3_strict_evaluation():
    """Сценарий 4: Строгая оценка (низкая температура)"""

    custom_model = create_proxy_llm(
        model="gpt-4o-mini",
        api_key="sk-proxy-your-key",
        base_url="http://5.11.83.110:8000"
    )

    rag_connector = RAGConnector(
        endpoint_url="http://5.11.83.110:8002/api/v1/chat/",
        api_key="rag-api-key",
        timeout=30
    )

    excel_path = "data/evaluation_dataset.xlsx"

    metrics_to_use = [
        'answer_relevancy',
        'faithfulness'
    ]

    results = await evaluate_rag_from_excel(
        excel_path=excel_path,
        rag_connector=rag_connector,
        metrics_list=metrics_to_use,
        model=custom_model,
        threshold=0.8,  # Высокий порог
        temperature=0.1,  # STRICT: все вердикты важны
        sleep_time=0.1,
        verbose=True,
        show_dashboard=True,
        session_name="Strict Evaluation"
    )

    return results


async def scenario_4_lenient_evaluation():
    """Сценарий 5: Мягкая оценка (высокая температура)"""

    custom_model = create_proxy_llm(
        model="gpt-4o-mini",
        api_key="sk-proxy-your-key",
        base_url="http://5.11.83.110:8000"
    )

    rag_connector = RAGConnector(
        endpoint_url="http://5.11.83.110:8002/api/v1/chat/",
        api_key="rag-api-key",
        timeout=30
    )

    excel_path = "data/evaluation_dataset.xlsx"

    metrics_to_use = [
        'answer_relevancy',
        'faithfulness'
    ]

    results = await evaluate_rag_from_excel(
        excel_path=excel_path,
        rag_connector=rag_connector,
        metrics_list=metrics_to_use,
        model=custom_model,
        threshold=0.6,  # Низкий порог
        temperature=1.0,  # LENIENT: фокус на хороших вердиктах
        sleep_time=0.1,
        verbose=True,
        show_dashboard=True,
        session_name="Lenient Evaluation"
    )

    return results


if __name__ == "__main__":

    asyncio.run(scenario_3_strict_evaluation())
