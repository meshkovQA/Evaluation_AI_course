"""
DeepEval мультимодальные метрики для оценки изображений

Метрики на основе GPT-4o (или другой MLLM):
- ImageCoherence: связность изображения с текстом
- ImageHelpfulness: полезность изображения для понимания
- TextToImageMetric: качество генерации изображения по prompt
- MultimodalAnswerRelevancy: релевантность RAG ответа с изображениями

Все метрики возвращают score 0-1 (чем выше, тем лучше)
"""

from pathlib import Path
from typing import Union, Optional, List

# DeepEval imports
from deepeval.test_case import MLLMTestCase, MLLMImage
from deepeval.metrics import (
    ImageCoherenceMetric,
    ImageHelpfulnessMetric,
    TextToImageMetric,
    MultimodalAnswerRelevancyMetric
)
from deepeval import evaluate

# Импортируем ProxyMLLM из локального файла
# Поддержка запуска и как модуля, и напрямую
try:
    from .deepeval_custom_llm import ProxyMLLM, create_proxy_model
except ImportError:
    from deepeval_custom_llm import ProxyMLLM, create_proxy_model


def _create_mllm_image(path: Union[str, Path]) -> MLLMImage:
    """
    Создаёт MLLMImage объект из пути к файлу.

    Args:
        path: Путь к изображению (локальный или URL)

    Returns:
        MLLMImage объект для DeepEval
    """
    path_str = str(path)

    if path_str.startswith(("http://", "https://")):
        return MLLMImage(url=path_str, local=False)
    else:
        abs_path = str(Path(path).resolve())
        return MLLMImage(url=abs_path, local=True)


def _build_actual_output(items: List) -> List:
    """
    Преобразует список элементов в actual_output для DeepEval.
    Строки-пути к файлам конвертируются в MLLMImage.
    """
    actual_output = []
    for item in items:
        if isinstance(item, str):
            if Path(item).exists():
                actual_output.append(_create_mllm_image(item))
            else:
                actual_output.append(item)
        elif isinstance(item, MLLMImage):
            actual_output.append(item)
        else:
            actual_output.append(item)
    return actual_output


def evaluate_image_coherence(
    input_prompt: str,
    actual_output: List,
    threshold: float = 0.5,
    model: str = "gpt-4o-mini"
) -> dict:

    test_case = MLLMTestCase(
        input=[input_prompt],
        actual_output=_build_actual_output(actual_output)
    )

    custom_model = create_proxy_model(model=model)
    metric = ImageCoherenceMetric(
        threshold=threshold,
        model=custom_model
    )

    res = evaluate(test_cases=[test_case], metrics=[metric])

    return res


def evaluate_image_helpfulness(
    input_prompt: str,
    actual_output: List,
    threshold: float = 0.5,
    model: str = "gpt-4o-mini"
) -> dict:

    test_case = MLLMTestCase(
        input=[input_prompt],
        actual_output=_build_actual_output(actual_output)
    )

    custom_model = create_proxy_model(model=model)
    metric = ImageHelpfulnessMetric(
        threshold=threshold,
        model=custom_model
    )

    res = evaluate(test_cases=[test_case], metrics=[metric])

    return res


def evaluate_text_to_image(
    prompt: str,
    generated_image_path: Union[str, Path],
    threshold: float = 0.5,
    model: str = "gpt-4o-mini"
) -> dict:

    test_case = MLLMTestCase(
        input=[prompt],
        actual_output=[_create_mllm_image(generated_image_path)]
    )

    custom_model = create_proxy_model(model=model)
    metric = TextToImageMetric(
        threshold=threshold,
        model=custom_model
    )

    res = evaluate(test_cases=[test_case], metrics=[metric])
    return res


def evaluate_multimodal_relevancy(
    question: str,
    answer_with_images: List,
    threshold: float = 0.5,
    model: str = "gpt-4o-mini"
) -> dict:

    test_case = MLLMTestCase(
        input=[question],
        actual_output=_build_actual_output(answer_with_images),

    )

    custom_model = create_proxy_model(model=model)
    metric = MultimodalAnswerRelevancyMetric(
        threshold=threshold,
        model=custom_model
    )

    res = evaluate(test_cases=[test_case], metrics=[metric])

    return res


# Тестирование при запуске напрямую
if __name__ == "__main__":
    print("=" * 50)
    print("Тестирование DeepEval мультимодальных метрик")
    print("=" * 50)

    # Используем готовые sample_images
    sample_dir = Path(__file__).parent.parent / "sample_images"

    original = sample_dir / "original.png"
    edited = sample_dir / "edited.png"
    different = sample_dir / "different.png"

    if not original.exists():
        print(f"\n⚠️  Тестовые изображения не найдены в {sample_dir}")
        print("   Добавьте изображения в папку sample_images/")
        exit(1)

    print(f"\n📁 Используем изображения из {sample_dir}")

    # ---------------------------------------------------
    # 1. TextToImageMetric
    # ---------------------------------------------------
    print("\n" + "-" * 40)
    print("1. TextToImageMetric")
    print("-" * 40)

    prompt = "a man with glasses working on a laptop with code on monitors in an office"

    result = evaluate_text_to_image(
        prompt=prompt,
        generated_image_path=original
    )

    # ---------------------------------------------------
    # 2. ImageCoherence
    # ---------------------------------------------------
    print("\n" + "-" * 40)
    print("2. ImageCoherence")
    print("-" * 40)

    input_prompt = "Show me a software developer at work"
    actual_output = [
        str(original)
    ]

    result = evaluate_image_coherence(
        input_prompt=input_prompt,
        actual_output=actual_output
    )

    # ---------------------------------------------------
    # 3. ImageHelpfulness
    # ---------------------------------------------------
    print("\n" + "-" * 40)
    print("3. ImageHelpfulness")
    print("-" * 40)

    input_prompt = "Show me what a programmer's workspace looks like"
    actual_output = [
        str(edited)
    ]

    result = evaluate_image_helpfulness(
        input_prompt=input_prompt,
        actual_output=actual_output
    )

    # ---------------------------------------------------
    # 4. MultimodalAnswerRelevancy
    # ---------------------------------------------------
    print("\n" + "-" * 40)
    print("4. MultimodalAnswerRelevancy")
    print("-" * 40)

    question = "What does a software developer's workspace look like?"
    answer_with_images = [
        "A software developer's workspace typically includes a computer or laptop, multiple monitors displaying code, and other tech accessories.",
        str(original),
        "Here's an edited image showing a more organized and modern workspace.",
        str(edited)
    ]

    result = evaluate_multimodal_relevancy(
        question=question,
        answer_with_images=answer_with_images
    )
