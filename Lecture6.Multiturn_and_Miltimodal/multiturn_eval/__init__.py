"""
Multiturn Evaluation Module для DeepEval
Интерактивный CLI чат с оценкой многоходовых бесед

Автономный модуль - не зависит от внешних файлов проекта.
"""

from .conversation_storage import ConversationStorage
from .evaluators.deepeval.multiturn_evaluator import (
    create_metrics,
    conversation_to_test_case,
    run_evaluation,
    DEFAULT_CHATBOT_ROLE
)
from .agent_connector import AgentConnector
from .evaluators.deepeval.deepeval_custom_llm import ProxyLLM, create_proxy_model

__all__ = [
    'ConversationStorage',
    'create_metrics',
    'conversation_to_test_case',
    'run_evaluation',
    'DEFAULT_CHATBOT_ROLE',
    'AgentConnector',
    'ProxyLLM',
    'create_proxy_model'
]
