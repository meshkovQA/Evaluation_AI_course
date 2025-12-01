"""
CLIP Score метрики для оценки изображений

CLIP Score измеряет семантическое сходство между:
- Изображением и текстом (image-text similarity)
- Двумя изображениями (image-image similarity)

Диапазон значений: 0-100 (чем выше, тем лучше)
"""
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_loader import preprocess_for_clip, load_image  # noqa: E402


import torch  # noqa: E402
from torchmetrics.multimodal.clip_score import CLIPScore  # noqa: E402
from typing import Union, Optional  # noqa: E402


# Добавляем родительскую директорию в путь (для запуска напрямую)


# Глобальный кэш для модели (ленивая инициализация)
_clip_model: Optional[CLIPScore] = None


def _get_clip_model(model_name: str = "openai/clip-vit-base-patch16") -> CLIPScore:

    global _clip_model

    if _clip_model is None:
        print(f"📥 Загрузка модели CLIP: {model_name}")
        _clip_model = CLIPScore(model_name_or_path=model_name)
        # Явно устанавливаем CPU
        _clip_model = _clip_model.to("cpu")
        print("✅ Модель CLIP загружена")

    return _clip_model


def compute_clip_score_text(
    image_path: Union[str, Path],
    text: str,
    model_name: str = "openai/clip-vit-base-patch16"
) -> float:

    # Получаем модель
    metric = _get_clip_model(model_name)

    # Предобработка изображения
    image_tensor = preprocess_for_clip(image_path)

    # Вычисляем score
    with torch.no_grad():
        score = metric(image_tensor, text)

    return score.detach().item()


def compute_clip_score_images(
    image1_path: Union[str, Path],
    image2_path: Union[str, Path],
    model_name: str = "openai/clip-vit-base-patch16"
) -> float:

    metric = _get_clip_model(model_name)

    # Предобработка изображений
    image1_tensor = preprocess_for_clip(image1_path)
    image2_tensor = preprocess_for_clip(image2_path)

    # Вычисляем score между изображениями
    with torch.no_grad():
        score = metric(image1_tensor, image2_tensor)

    return score.detach().item()


def compute_clip_score_batch(
    image_paths: list,
    texts: list,
    model_name: str = "openai/clip-vit-base-patch16"
) -> list:

    if len(image_paths) != len(texts):
        raise ValueError("Количество изображений и текстов должно совпадать")

    scores = []
    for img_path, text in zip(image_paths, texts):
        score = compute_clip_score_text(img_path, text, model_name)
        scores.append(score)

    return scores


def interpret_clip_score(score: float) -> str:

    if score >= 30:
        return "Отличное соответствие"
    elif score >= 25:
        return "Хорошее соответствие"
    elif score >= 20:
        return "Умеренное соответствие"
    elif score >= 15:
        return "Слабое соответствие"
    else:
        return "Очень слабое соответствие"


# Тестирование при запуске напрямую
if __name__ == "__main__":
    print("=" * 50)
    print("Тестирование CLIP Score метрик")
    print("=" * 50)

    # Используем готовые sample_images
    sample_dir = Path(__file__).parent.parent / "sample_images"

    original = sample_dir / "original.png"
    edited = sample_dir / "edited.png"
    different = sample_dir / "different.png"

    if not original.exists():
        print(f"\n⚠️  Тестовые изображения не найдены в {sample_dir}")
        print("   Запустите: python create_sample_images.py")
        exit(1)

    print(f"\n📁 Используем изображения из {sample_dir}")

    # ---------------------------------------------------
    # 1. CLIP Score (image-text)
    # ---------------------------------------------------
    print("\n1. CLIP Score (image-text):")
    print("-" * 40)

    # Тест: правильное описание
    prompt_correct = "a man with glasses working on a laptop with code on monitors"
    score_match = compute_clip_score_text(original, prompt_correct)
    print(f"   Original + '{prompt_correct}'")
    print(f"   Score: {score_match:.2f} - {interpret_clip_score(score_match)}")

    # Тест: неправильное описание
    prompt_wrong = "a cat sleeping on a sofa"
    score_wrong = compute_clip_score_text(original, prompt_wrong)
    print(f"\n   Original + '{prompt_wrong}'")
    print(f"   Score: {score_wrong:.2f} - {interpret_clip_score(score_wrong)}")

    # ---------------------------------------------------
    # 2. CLIP Score (image-image)
    # ---------------------------------------------------
    print("\n2. CLIP Score (image-image):")
    print("-" * 40)

    # Похожие изображения (тот же человек, немного другой ракурс)
    score_similar = compute_clip_score_images(original, edited)
    print(f"   Original vs Edited (тот же человек): {score_similar:.2f}")

    # Разные изображения (разные люди)
    score_different = compute_clip_score_images(original, different)
    print(f"   Original vs Different (другой человек): {score_different:.2f}")

    # Идентичные
    score_same = compute_clip_score_images(original, original)
    print(f"   Идентичные изображения: {score_same:.2f}")

    print("\n✅ Тестирование завершено!")
    print("\n📝 Примечание:")
    print("   CLIP Score 0-100, чем выше - тем лучше соответствие")
