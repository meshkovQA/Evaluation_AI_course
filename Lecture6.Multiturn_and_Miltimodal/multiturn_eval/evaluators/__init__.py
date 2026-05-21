"""
Модуль оценки многоходовых бесед
Поддерживает DeepEval и Opik фреймворки
"""

# Импорт DeepEval
from .deepeval import run_evaluation as run_deepeval_evaluation
from .deepeval import DEFAULT_CHATBOT_ROLE
from .deepeval import create_proxy_model

# Пробуем импортировать Opik
try:
    from .opik import run_evaluation as run_opik_evaluation
    HAS_OPIK = True
except ImportError:
    HAS_OPIK = False
    run_opik_evaluation = None


def run_evaluation(
    conversations,
    framework: str = "deepeval",
    **kwargs
):
    """
    Универсальная функция оценки с выбором фреймворка

    Args:
        conversations: список диалогов
        framework: "deepeval" или "opik"
        **kwargs: дополнительные параметры для выбранного фреймворка

    Returns:
        Результаты оценки
    """
    if framework == "deepeval":
        return run_deepeval_evaluation(conversations, **kwargs)
    elif framework == "opik":
        if not HAS_OPIK:
            raise ImportError("Opik не установлен. Установите: pip install opik")
        return run_opik_evaluation(conversations, **kwargs)
    else:
        raise ValueError(f"Неизвестный фреймворк: {framework}. Используйте 'deepeval' или 'opik'")