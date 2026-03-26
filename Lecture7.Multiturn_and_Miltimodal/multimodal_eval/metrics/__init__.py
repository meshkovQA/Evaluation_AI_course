"""
Метрики для мультимодальной оценки изображений
"""

from .clip_score import compute_clip_score_text, compute_clip_score_images
from .lpips_metric import compute_lpips
from .deepeval_custom_llm import ProxyLLM, ProxyMLLM, create_proxy_model, create_proxy_llm
from .deepeval_multimodal import (
    evaluate_image_coherence,
    evaluate_image_helpfulness,
    evaluate_text_to_image,
    evaluate_multimodal_relevancy
)

__all__ = [
    # CLIP Score
    "compute_clip_score_text",
    "compute_clip_score_images",
    # LPIPS
    "compute_lpips",
    # DeepEval LLM/MLLM
    "ProxyLLM",
    "ProxyMLLM",
    "create_proxy_model",
    "create_proxy_llm",
    # DeepEval метрики
    "evaluate_image_coherence",
    "evaluate_image_helpfulness",
    "evaluate_text_to_image",
    "evaluate_multimodal_relevancy"
]
