"""
DeepEval оценка многоходовых бесед
"""

from .multiturn_evaluator import (
    run_evaluation,
    create_metrics,
    conversation_to_test_case,
    DEFAULT_CHATBOT_ROLE
)
from .deepeval_custom_llm import ProxyLLM, create_proxy_model