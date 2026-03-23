"""
Интеграционный тест для оценки агента с eval_lib
Метрики: AnswerRelevancy, ToolCorrectness, TaskSuccessRate
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio  # noqa: E402
import time  # noqa: E402
from typing import List, Optional  # noqa: E402
from eval_lib import (
    evaluate,
    EvalTestCase,
    AnswerRelevancyMetric,
    ToolCorrectnessMetric,
    TaskSuccessRateMetric,
    CustomLLMClient
)  # noqa: E402
from dataset_parser import DatasetParser  # noqa: E402
from agent_connector import AgentConnector  # noqa: E402
from custom_proxy_llm import create_proxy_llm  # noqa: E402


async def evaluate_agent_with_eval_lib(
    excel_path: str,
    agent_connector: AgentConnector,
    metrics_list: List[str],
    urls: Optional[List[str]] = None,
    model: str | CustomLLMClient = "gpt-4o-mini",
    threshold: float = 0.7,
    temperature: float = 0.5,
    sleep_time: float = 0.5,
    verbose: bool = True,
    show_dashboard: bool = False,
    session_name: str = "Agent Evaluation"
):
    """
    Оценка агента с использованием eval_lib

    Args:
        excel_path: путь к датасету
        agent_connector: коннектор к агенту
        metrics_list: список метрик для оценки
        urls: список URL для передачи агенту
        model: модель для метрик
        threshold: порог для метрик
        temperature: температура для LLM
        sleep_time: задержка между запросами
        verbose: детальный вывод
        show_dashboard: показать dashboard
        session_name: название сессии
    """

    print("\n" + "=" * 70)
    print(f"🚀 {session_name.upper()}")
    print("=" * 70)

    # ============= ШАГ 1: Загрузка датасета =============
    print("\n📂 Шаг 1: Загрузка датасета...")

    parser = DatasetParser()
    df = parser.load_dataset(excel_path)

    if df is None:
        print("❌ Ошибка загрузки датасета")
        return None

    # Показываем информацию
    info = parser.validate_dataset(df)
    print(f"\n📊 Информация о датасете:")
    print(f"   • Всего строк: {info['total_rows']}")
    print(f"   • Валидных пар: {info['valid_pairs']}")
    if info['has_expected_tools_column']:
        print(f"   • Среднее количество тулов: {info['avg_tools_count']:.1f}")
        print(f"   • Уникальных тулов: {len(info['unique_tools'])}")

    # Превью
    parser.preview_dataset(df, n=2)

    # ============= ШАГ 2: Извлечение данных =============
    print("\n📝 Шаг 2: Извлечение данных из датасета...")

    pairs = parser.get_question_response_pairs(df)
    print(f"✅ Получено {len(pairs)} пар")

    if urls:
        print(f"🔗 URLs для агента: {urls}")

    # ============= ШАГ 3: Запросы к агенту =============
    print("\n🤖 Шаг 3: Получение ответов от агента...")

    test_cases = []

    for i, pair in enumerate(pairs, 1):
        question = pair['question']
        expected_response = pair['expected_response']
        expected_tools = pair['expected_tools']

        print(f"\n[{i}/{len(pairs)}] {question[:60]}...")

        # Запрос к агенту с URLs
        response = agent_connector.query(question, urls=urls)

        if response.get('error'):
            print(f"   ❌ {response['error']}")
            continue

        # Извлекаем данные
        answer = response.get('output', '')
        tools_used = response.get('tools_used', [])

        print(f"   ✅ Ответ: {answer[:50]}...")
        print(f"   🔧 Тулы: {tools_used}")

        # Создаем EvalTestCase
        test_case = EvalTestCase(
            input=question,
            actual_output=answer,
            expected_output=expected_response,
            tools_called=tools_used,  # список строк
            expected_tools=expected_tools  # список строк
        )

        test_cases.append(test_case)

        if i < len(pairs):
            time.sleep(sleep_time)

    print(f"\n✅ Создано {len(test_cases)} тест кейсов")

    # ============= ШАГ 4: Создание метрик =============
    print("\n📊 Шаг 4: Создание метрик...")
    print(f"   Метрики: {metrics_list}")
    print(f"   Порог: {threshold}")
    print(f"   Температура: {temperature}")

    # Определяем тип модели
    model_name = model.get_model_name() if isinstance(
        model, CustomLLMClient) else model
    print(f"   Модель: {model_name}")

    metrics = []
    metric_classes = {
        'answer_relevancy': AnswerRelevancyMetric,
        'tool_correctness': ToolCorrectnessMetric,
        'task_success_rate': TaskSuccessRateMetric
    }

    for metric_name in metrics_list:
        if metric_name == 'tool_correctness':
            metric = ToolCorrectnessMetric(
                threshold=threshold,
                verbose=verbose,
                exact_match=True,
                check_ordering=True
            )
            metrics.append(metric)
            print(f"   ✅ ToolCorrectnessMetric")
        elif metric_name in metric_classes:
            metric = metric_classes[metric_name](
                model=model,
                threshold=threshold,
                temperature=temperature,
                verbose=verbose
            )
            metrics.append(metric)
            print(f"   ✅ {metric_name}")

    print(f"\n✅ Создано {len(metrics)} метрик")

    # ============= ШАГ 5: Запуск оценки =============
    print("\n🧪 Шаг 5: Запуск оценки...")
    print(f"   Тест кейсов: {len(test_cases)}")
    print(f"   Метрик: {len(metrics)}")
    print("=" * 70)

    try:
        # Запускаем оценку
        results = await evaluate(
            test_cases=test_cases,
            metrics=metrics,
            verbose=verbose,
            show_dashboard=show_dashboard,
            session_name=session_name
        )

        print("\n" + "=" * 70)
        print("✅ ОЦЕНКА ЗАВЕРШЕНА")
        print("=" * 70)

        return results

    except Exception as e:
        print(f"\n❌ Ошибка при оценке: {e}")
        import traceback
        traceback.print_exc()
        return None


async def scenario_1():
    """Сценарий 2: Proxy модель"""

    print("\n" + "=" * 70)
    print("📋 СЦЕНАРИЙ 2: Оценка с Proxy моделью")
    print("=" * 70)

    # Агент
    agent = AgentConnector(
        endpoint_url="http://5.11.83.110:8004/ask",
        api_key="80456142-5441-4469-b97f-1d72b7802a93",
        user_id="AleksM",
        session_id="faecb783-d996-49ee-97a0-f13805f63a52"
    )

    # Proxy модель
    proxy_model = create_proxy_llm(
        model="gpt-4o-mini",
        api_key="sk-proxy-maEZp5Yp0-h9nOHDZXtoZPql5VRW3CqTqakKOQgsQtQ",
        base_url="http://5.11.83.110:8000"
    )

    # Датасет
    excel_path = "data/evaluation_dataset.xlsx"

    # URLs для агента
    test_urls = [
        "https://www.rentalads.com/apartments-for-rent/ny/new-york/"
    ]

    # Метрики
    metrics_to_use = [
        'answer_relevancy',
        'tool_correctness',
        'task_success_rate'
    ]

    # Запуск оценки
    results = await evaluate_agent_with_eval_lib(
        excel_path=excel_path,
        agent_connector=agent,
        metrics_list=metrics_to_use,
        urls=test_urls,
        model=proxy_model,  # или просто "gpt-4o-mini"
        threshold=0.7,
        temperature=0.5,
        sleep_time=0.5,
        verbose=True,
        show_dashboard=True,
        session_name="Agent Evaluation - Proxy"
    )

    return results


if __name__ == "__main__":

    asyncio.run(scenario_1())
