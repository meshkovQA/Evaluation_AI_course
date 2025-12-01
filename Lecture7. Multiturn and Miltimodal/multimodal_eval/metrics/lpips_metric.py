"""
LPIPS (Learned Perceptual Image Patch Similarity) метрика

LPIPS измеряет перцептуальное сходство между двумя изображениями,
используя глубинные признаки из предобученных сетей.

Диапазон значений: 0-1 (чем НИЖЕ, тем более похожи изображения)
- 0.0 = идентичные изображения
- ~0.5 = умеренно различные изображения
- >0.7 = очень разные изображения
"""
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
# Добавляем родительскую директорию в путь (для запуска напрямую)
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_loader import preprocess_for_lpips  # noqa: E402

from typing import Union, Optional  # noqa: E402
import lpips  # noqa: E402
import torch  # noqa: E402


# Глобальный кэш для модели (ленивая инициализация)
_lpips_model: Optional[lpips.LPIPS] = None
_current_net: Optional[str] = None


def _get_lpips_model(net: str = "squeeze") -> lpips.LPIPS:

    global _lpips_model, _current_net

    if _lpips_model is None or _current_net != net:
        print(f"📥 Загрузка модели LPIPS (сеть: {net})")
        _lpips_model = lpips.LPIPS(net=net)
        _lpips_model.eval()  # Режим inference
        _current_net = net
        print("✅ Модель LPIPS загружена")

    return _lpips_model


def compute_lpips(
    image1_path: Union[str, Path],
    image2_path: Union[str, Path],
    net: str = "squeeze"
) -> float:

    # Получаем модель
    loss_fn = _get_lpips_model(net)

    # Предобработка изображений
    img1_tensor = preprocess_for_lpips(image1_path)
    img2_tensor = preprocess_for_lpips(image2_path)

    # Вычисляем расстояние
    with torch.no_grad():
        distance = loss_fn(img1_tensor, img2_tensor)

    return distance.item()


def compute_lpips_batch(
    reference_paths: list,
    comparison_paths: list,
    net: str = "squeeze"
) -> list:

    if len(reference_paths) != len(comparison_paths):
        raise ValueError(
            "Количество эталонных и сравниваемых изображений должно совпадать")

    distances = []
    for ref_path, cmp_path in zip(reference_paths, comparison_paths):
        dist = compute_lpips(ref_path, cmp_path, net)
        distances.append(dist)

    return distances


def interpret_lpips(distance: float) -> str:

    if distance < 0.05:
        return "Практически идентичные изображения"
    elif distance < 0.1:
        return "Очень похожие изображения"
    elif distance < 0.2:
        return "Похожие изображения с небольшими различиями"
    elif distance < 0.4:
        return "Заметные различия между изображениями"
    elif distance < 0.6:
        return "Значительные различия"
    else:
        return "Очень разные изображения"


def lpips_to_similarity(distance: float) -> float:

    return 1.0 - min(distance, 1.0)


# Тестирование при запуске напрямую
if __name__ == "__main__":
    print("=" * 50)
    print("Тестирование LPIPS метрики")
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
    # 1. Идентичные изображения
    # ---------------------------------------------------
    print("\n1. Сравнение идентичных изображений:")
    print("-" * 40)
    dist_identical = compute_lpips(original, original)
    print(f"   LPIPS Distance: {dist_identical:.4f}")
    print(f"   Интерпретация: {interpret_lpips(dist_identical)}")
    print(f"   Similarity: {lpips_to_similarity(dist_identical):.2%}")

    # ---------------------------------------------------
    # 2. Похожие изображения (тот же человек, другой ракурс)
    # ---------------------------------------------------
    print("\n2. Похожие изображения (original vs edited - тот же человек):")
    print("-" * 40)
    dist_similar = compute_lpips(original, edited)
    print(f"   LPIPS Distance: {dist_similar:.4f}")
    print(f"   Интерпретация: {interpret_lpips(dist_similar)}")
    print(f"   Similarity: {lpips_to_similarity(dist_similar):.2%}")

    # ---------------------------------------------------
    # 3. Разные изображения (разные люди)
    # ---------------------------------------------------
    print("\n3. Разные изображения (original vs different - другой человек):")
    print("-" * 40)
    dist_different = compute_lpips(original, different)
    print(f"   LPIPS Distance: {dist_different:.4f}")
    print(f"   Интерпретация: {interpret_lpips(dist_different)}")
    print(f"   Similarity: {lpips_to_similarity(dist_different):.2%}")

    print("\n✅ Тестирование завершено!")
    print("\n📝 Примечание:")
    print("   LPIPS 0-1, чем НИЖЕ - тем более похожи изображения")
