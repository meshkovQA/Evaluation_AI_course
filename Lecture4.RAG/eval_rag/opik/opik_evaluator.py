# opik_evaluator.py

from typing import List, Dict, Union
import opik
from opik.evaluation.metrics import (
    AnswerRelevance,
    Hallucination,
    ContextPrecision,
    ContextRecall
)
from opik.evaluation import evaluate
from opik import Opik
from opik_custom_llm import create_opik_proxy_model, OpikProxyLLM


def create_metrics(
    model: Union[str, OpikProxyLLM] = "gpt-4o-mini",
    metrics_list: List[str] = None,
    threshold: float = 0.7
):
    """
    Создать метрики Opik

    Args:
        model: "gpt-4o-mini" ИЛИ OpikProxyLLM экземпляр
        metrics_list: ['answer_relevance', 'hallucination', 'context_precision', 'context_recall']
        threshold: Порог для метрик

    Returns:
        Список метрик Opik
    """
    if metrics_list is None:
        metrics_list = ['answer_relevance', 'hallucination']

    METRICS = {
        'answer_relevance': AnswerRelevance,
        'hallucination': Hallucination,
        'context_precision': ContextPrecision,
        'context_recall': ContextRecall
    }

    metrics = []
    for metric_name in metrics_list:
        if metric_name in METRICS:
            kwargs = {"name": metric_name}
            if model is not None:
                kwargs["model"] = model
            metric = METRICS[metric_name](**kwargs)
            metrics.append(metric)

    return metrics


def simple_rag_task(item: Dict) -> Dict:
    """
    Простая task функция для эксперимента
    В реальности здесь был бы вызов вашей RAG-системы
    """
    return {
        "input": item["input"],
        "output": item["output"],
        "context": item.get("context", []),
        "expected_output": item.get("expected_output", "")
    }


# if __name__ == "__main__":

#     # Настройка Opik
#     opik.configure(use_local=False)  # Или use_local=True для локального
#     client = Opik()

#     # Тестовые данные
#     test_input = "What if these shoes don't fit?"
#     test_output = "We offer a 30-day full refund at no extra cost."
#     test_context = [
#         "All customers are eligible for a 30 day full refund at no extra cost."]

#     # Тестовые примеры
#     test_cases = [
#         {
#             "input": "What if these shoes don't fit?",
#             "output": "We offer a 30-day full refund at no extra cost.",
#             "expected_output": "You are eligible for a 30 day full refund at no extra cost.",
#             "context": ["All customers are eligible for a 30 day full refund at no extra cost."]
#         },
#         {
#             "input": "What is your return policy?",
#             "output": "You can return within 30 days for a full refund.",
#             "expected_output": "We have a 30-day return policy with full refund.",
#             "context": ["Our return policy allows 30 days for returns with full refund."]
#         }
#     ]

#     # Создаем прокси модель
#     proxy_model = create_opik_proxy_model(
#         model="gpt-4o-mini",
#         api_key="sk-proxy-your-key",
#         base_url="http://5.11.83.110:8000"
#     )

    # # ============= ПРИМЕР 1: Прямое использование метрик =============
    # print("📊 Пример 1: Прямое использование метрик")

    # # Создаем метрики
    # answer_relevance = AnswerRelevance()
    # hallucination = Hallucination()

    # # Оцениваем
    # score1 = answer_relevance.score(
    #     input=test_input,
    #     output=test_output,
    #     context=test_context
    # )

    # score2 = hallucination.score(
    #     input=test_input,
    #     output=test_output,
    #     context=test_context
    # )

    # print(f"\n✅ Answer Relevance Score: {score1.value:.3f}")
    # print(f"   Reason: {score1.reason}")
    # print(f"\n✅ Hallucination Score: {score2.value:.3f}")
    # print(f"   Reason: {score2.reason}")

    # # ============= ПРИМЕР 2: С Datasets и Experiments =============
    # print("\n" + "="*60)
    # print("📊 Пример 2: Полноценный эксперимент с датасетом")

    # # Создаем датасет (или получаем существующий)
    # dataset_name = "simple_test_dataset_v3"

    # # Создаем датасет с данными сразу
    # dataset = client.create_dataset(
    #     name=dataset_name,
    #     description="Датасет для оценки RAG-системы"
    # )

    # # Правильный способ добавления данных в датасет
    # dataset.insert(test_cases)

    # print(f"✅ Создан датасет '{dataset_name}' с {len(test_cases)} примерами")

    # # Создаем метрики для эксперимента
    # experiment_metrics = create_metrics(
    #     metrics_list=['answer_relevance', 'hallucination', 'context_precision']
    # )

    # # Запускаем эксперимент
    # print("\n🧪 Запуск эксперимента...")
    # result = evaluate(
    #     dataset=dataset,
    #     task=simple_rag_task,
    #     scoring_metrics=experiment_metrics,
    #     experiment_name="Simple_RAG_Test_v3"
    # )

    # print("\n✅ Все примеры выполнены!")
    # print("📊 Перейдите на https://www.comet.com/opik для просмотра результатов")

    # # ============= ПРИМЕР 3: С ПРОКСИ =============
    # print("\n" + "="*60)
    # print("📊 Пример 2: Кастомная прокси модель")

    # # Создаем метрики с прокси моделью
    # metrics = create_metrics(
    #     model=proxy_model,  # ← Кастомная прокси модель
    #     metrics_list=['answer_relevance', 'hallucination']
    # )

    # # Оцениваем
    # for metric in metrics:
    #     score = metric.score(
    #         input=test_input,
    #         output=test_output,
    #         context=test_context
    #     )
    #     print(f"\n✅ {metric.name} (proxy) Score: {score.value:.3f}")
    #     print(f"   Reason: {score.reason}")

    # ============= ПРИМЕР 4: С Datasets и Experiments (с прокси) =============
    # print("\n" + "="*60)
    # print("📊 Пример 4: Эксперимент с датасетом и прокси")

    # # Получаем существующий датасет
    # dataset = client.get_dataset(name="simple_test_dataset_v2")

    # Имя датасета
    # import time
    # dataset_name = f"test_{int(time.time())}"

    # # Создаем датасет
    # dataset = client.create_dataset(
    #     name=dataset_name,
    #     description="Датасет для оценки RAG через прокси"
    # )
    # dataset.insert(test_cases)
    # print(f"✅ Создан датасет '{dataset_name}' с {len(test_cases)} примерами")

    # # Создаем метрики с прокси для эксперимента
    # experiment_metrics = create_metrics(
    #     model=proxy_model,  # ← Используем прокси модель
    #     metrics_list=['answer_relevance', 'hallucination', 'context_precision']
    # )

    # # Запускаем эксперимент
    # print("\n🧪 Запуск эксперимента с прокси...")
    # result = evaluate(
    #     dataset=dataset,
    #     task=simple_rag_task,
    #     scoring_metrics=experiment_metrics,
    #     experiment_name="Proxy_RAG_Test_v4"
    # )
