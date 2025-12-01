"""
Оценка многоходовых бесед с использованием DeepEval метрик
"""

from typing import List, Dict, Union, Optional
from deepeval.test_case import ConversationalTestCase, Turn, ToolCall
from deepeval.metrics import (
    RoleAdherenceMetric,
    KnowledgeRetentionMetric,
    ConversationCompletenessMetric
)
from deepeval import evaluate

# Пробуем импортировать TurnRelevancyMetric (может быть недоступен в старых версиях)
try:
    from deepeval.metrics import TurnRelevancyMetric
    HAS_TURN_RELEVANCY = True
except ImportError:
    HAS_TURN_RELEVANCY = False
    print("Предупреждение: TurnRelevancyMetric недоступен в вашей версии deepeval")

from .deepeval_custom_llm import ProxyLLM


# Роль чатбота по умолчанию (используется для RoleAdherenceMetric)
DEFAULT_CHATBOT_ROLE = """AI-ассистент по поиску и анализу недвижимости.
Помогает пользователям находить квартиры, фильтровать по различным критериям
(цена, район, количество комнат), конвертировать валюты и предоставлять
статистику по ценам. Отвечает вежливо, информативно и на русском языке."""


def create_metrics(
    model: Union[str, ProxyLLM],
    threshold: float = 0.5,
    include_reason: bool = True
) -> List:
    """
    Создаёт список метрик для оценки многоходовых бесед
    """
    metrics = []

    # 1. TurnRelevancyMetric - релевантность каждого ответа
    if HAS_TURN_RELEVANCY:
        metrics.append(
            TurnRelevancyMetric(
                threshold=threshold,
                model=model,
                include_reason=include_reason
            )
        )

    # 2. RoleAdherenceMetric - соответствие роли
    metrics.append(
        RoleAdherenceMetric(
            threshold=threshold,
            model=model,
            include_reason=include_reason
        )
    )

    # 3. KnowledgeRetentionMetric - сохранение контекста
    metrics.append(
        KnowledgeRetentionMetric(
            threshold=threshold,
            model=model,
            include_reason=include_reason
        )
    )

    # 4. ConversationCompletenessMetric - полнота выполнения задач
    metrics.append(
        ConversationCompletenessMetric(
            threshold=threshold,
            model=model,
            include_reason=include_reason
        )
    )

    return metrics


def conversation_to_test_case(
    conv: Dict,
    chatbot_role: str = DEFAULT_CHATBOT_ROLE,
    expected_outcome: Optional[str] = None
) -> ConversationalTestCase:
    """
    Преобразует сохранённый диалог в ConversationalTestCase для DeepEval

    """
    turns = []

    for turn_data in conv.get('turns', []):
        role = turn_data.get('role')
        content = turn_data.get('content', '')

        # tools_called только для assistant
        tools_called = None
        if role == 'assistant':
            tools_list = turn_data.get('tools_called', [])
            if tools_list:
                tools_called = [ToolCall(name=tool_name)
                                for tool_name in tools_list]

        turn = Turn(
            role=role,
            content=content,
            tools_called=tools_called
        )
        turns.append(turn)

    return ConversationalTestCase(
        chatbot_role=chatbot_role,
        turns=turns,
        expected_outcome=expected_outcome
    )


def run_evaluation(
    conversations: List[Dict],
    model: Optional[Union[str, ProxyLLM]] = None,
    threshold: float = 0.5,
    chatbot_role: str = DEFAULT_CHATBOT_ROLE,
    verbose: bool = True
) -> None:
    """
    Запускает оценку списка диалогов

    """
    if not conversations:
        print("Нет диалогов для оценки")
        return

    if model is None:
        raise ValueError(
            "Модель для оценки не передана. Укажите model при вызове run_evaluation()")

    print("\n" + "=" * 70)
    print("🧪 ОЦЕНКА МНОГОХОДОВЫХ БЕСЕД (DeepEval)")
    print("=" * 70)

    # Создаём метрики
    print("\n📊 Инициализация метрик...")
    metrics = create_metrics(model, threshold=threshold)
    # metrics = create_metrics(model="gpt-4o-mini", threshold=threshold)

    metric_names = [type(m).__name__ for m in metrics]
    for name in metric_names:
        print(f"   • {name}")

    # Преобразуем диалоги в тест-кейсы
    print(f"\n📝 Преобразование {len(conversations)} диалогов в тест-кейсы...")
    test_cases = []
    for conv in conversations:
        test_case = conversation_to_test_case(conv, chatbot_role=chatbot_role)
        test_cases.append(test_case)

        if verbose:
            conv_id = conv.get('id', 'N/A')
            turns_count = len(conv.get('turns', []))
            print(f"   [{conv_id}] {turns_count} ходов")

    # Запускаем оценку
    print(f"\n🚀 Запуск оценки...")
    print("-" * 70)

    try:
        results = evaluate(
            test_cases=test_cases,
            metrics=metrics
        )

        print("\n" + "=" * 70)
        print("✅ ОЦЕНКА ЗАВЕРШЕНА")
        print("=" * 70)
        print("\n🌐 Подробные результаты: https://app.confident-ai.com")

    except Exception as e:
        print(f"\n❌ Ошибка при оценке: {e}")
        import traceback
        traceback.print_exc()
