"""
Оценка многоходовых бесед с использованием Opik метрик

Conversation LLM as a Judge Metrics:
- ConversationalCoherenceMetric - связность диалога
- SessionCompletenessQuality - выполнение целей пользователя
- UserFrustrationMetric - уровень фрустрации пользователя
"""

import os
from typing import List, Dict
import opik

# Импортируем метрики для бесед
from opik.evaluation.metrics import (
    ConversationalCoherenceMetric,
    SessionCompletenessQuality,
    UserFrustrationMetric
)


# Роль чатбота по умолчанию
DEFAULT_CHATBOT_ROLE = """AI-ассистент по поиску и анализу недвижимости.
Помогает пользователям находить квартиры, фильтровать по различным критериям
(цена, район, количество комнат), конвертировать валюты и предоставлять
статистику по ценам. Отвечает вежливо, информативно и на русском языке."""

# Название проекта по умолчанию для Opik
DEFAULT_PROJECT_NAME = "multiturn-evaluation"


def create_metrics(
    model: str = "gpt-4o-mini",
    include_reason: bool = True,
    window_size: int = 5
) -> List:
    """
    Создаёт список метрик для оценки многоходовых бесед

    """
    metrics = []

    # 1. ConversationalCoherenceMetric - связность диалога (LLM)
    metrics.append(
        ConversationalCoherenceMetric(
            model=model,
            window_size=window_size,
            include_reason=include_reason
        )
    )

    # 2. SessionCompletenessQuality - выполнение целей пользователя (LLM)
    metrics.append(
        SessionCompletenessQuality(
            model=model,
            include_reason=include_reason
        )
    )

    # 3. UserFrustrationMetric - уровень фрустрации (LLM)
    metrics.append(
        UserFrustrationMetric(
            model=model,
            include_reason=include_reason
        )
    )

    return metrics


def conversation_to_opik_format(conv: Dict) -> List[Dict[str, str]]:
    """
    Преобразует сохранённый диалог в формат Opik

    """
    conversation = []

    for turn_data in conv.get('turns', []):
        conversation.append({
            "role": turn_data.get('role'),
            "content": turn_data.get('content', '')
        })

    return conversation


def run_evaluation(
    conversations: List[Dict],
    model: str = "gpt-4o-mini",
    verbose: bool = True,
    project_name: str = DEFAULT_PROJECT_NAME,
    use_proxy: bool = False,
    proxy_api_key: str = None,
    proxy_base_url: str = None
) -> Dict:
    """
    Запускает оценку списка диалогов с метриками Opik

    """
    if not conversations:
        print("Нет диалогов для оценки")
        return {}

    # Настройка прокси через переменные окружения для litellm (который использует Opik)
    if use_proxy and proxy_api_key and proxy_base_url:
        os.environ["OPENAI_API_KEY"] = proxy_api_key
        os.environ["OPENAI_API_BASE"] = proxy_base_url
        print(f"   🔗 Прокси: {proxy_base_url}")

    # Инициализация Opik с указанием проекта
    opik.configure(use_local=False)

    # Устанавливаем проект по умолчанию для трейсов
    os.environ["OPIK_PROJECT_NAME"] = project_name

    print("\n" + "=" * 70)
    print("🧪 ОЦЕНКА МНОГОХОДОВЫХ БЕСЕД (Opik)")
    print(f"   📁 Проект: {project_name}")
    print(f"   🤖 Модель: {model}")
    print("=" * 70)

    # Создаём метрики с моделью (строка названия модели)
    print("\n📊 Инициализация метрик...")
    metrics = create_metrics(model=model)

    metric_names = [type(m).__name__ for m in metrics]
    for name in metric_names:
        print(f"   • {name}")

    # Преобразуем диалоги
    print(f"\n📝 Обработка {len(conversations)} диалогов...")

    all_results = []

    for conv in conversations:
        conv_id = conv.get('id', 'N/A')
        turns_count = len(conv.get('turns', []))

        if verbose:
            print(f"\n   [{conv_id}] {turns_count} ходов")

        # Преобразуем в формат Opik
        conversation = conversation_to_opik_format(conv)

        # Оцениваем каждой метрикой
        conv_results = {
            "id": conv_id,
            "turns_count": turns_count,
            "scores": {}
        }

        for metric in metrics:
            try:
                result = metric.score(conversation=conversation)
                conv_results["scores"][type(metric).__name__] = {
                    "value": result.value,
                    "reason": getattr(result, 'reason', None)
                }

                if verbose:
                    print(
                        f"      • {type(metric).__name__}: {result.value:.3f}")

            except Exception as e:
                conv_results["scores"][type(metric).__name__] = {
                    "value": None,
                    "error": str(e)
                }
                if verbose:
                    print(f"      • {type(metric).__name__}: ❌ {e}")

        all_results.append(conv_results)

    # Вычисляем средние значения
    print("\n" + "=" * 70)
    print("📈 СРЕДНИЕ ОЦЕНКИ:")
    print("-" * 70)

    avg_scores = {}
    for metric_name in metric_names:
        values = [
            r["scores"].get(metric_name, {}).get("value")
            for r in all_results
            if r["scores"].get(metric_name, {}).get("value") is not None
        ]
        if values:
            avg = sum(values) / len(values)
            avg_scores[metric_name] = avg
            print(f"   {metric_name}: {avg:.3f}")
        else:
            print(f"   {metric_name}: N/A")

    print("\n" + "=" * 70)
    print("✅ ОЦЕНКА ЗАВЕРШЕНА")
    print("=" * 70)
    print("\n🌐 Подробные результаты: https://www.comet.com/opik")

    return {
        "conversations": all_results,
        "averages": avg_scores
    }
