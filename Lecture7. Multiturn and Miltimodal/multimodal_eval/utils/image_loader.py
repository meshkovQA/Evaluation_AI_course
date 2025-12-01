"""
Утилиты для загрузки и предобработки изображений

Функции для работы с изображениями в контексте метрик CLIP Score и LPIPS.
"""

import torch
from PIL import Image
from torchvision import transforms
from typing import Union
from pathlib import Path


def load_image(path: Union[str, Path]) -> Image.Image:
    """
    Загружает изображение из файла.

    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Изображение не найдено: {path}")

    image = Image.open(path)

    # Конвертируем в RGB (на случай RGBA или других форматов)
    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def preprocess_for_clip(
    image: Union[str, Path, Image.Image],
    size: int = 224
) -> torch.Tensor:
    """
    Предобработка изображения для CLIP Score.

    CLIP ожидает изображения в формате uint8 [0-255] с размером 224x224.

    """
    # Загружаем если передан путь
    if isinstance(image, (str, Path)):
        image = load_image(image)

    # Трансформации для CLIP
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        # Конвертируем в uint8 [0-255] для torchmetrics CLIP
        transforms.Lambda(lambda x: (x * 255).to(torch.uint8))
    ])

    return transform(image)


def preprocess_for_lpips(
    image: Union[str, Path, Image.Image],
    size: int = 256,
    normalize: bool = True
) -> torch.Tensor:
    """
    Предобработка изображения для LPIPS.

    LPIPS ожидает изображения в диапазоне [-1, 1] (normalize=True)
    или [0, 1] (normalize=False, с флагом normalize=True в метрике).

    """
    # Загружаем если передан путь
    if isinstance(image, (str, Path)):
        image = load_image(image)

    # Базовые трансформации
    transform_list = [
        transforms.Resize((size, size)),
        transforms.ToTensor(),  # [0, 1]
    ]

    # Нормализация в [-1, 1] для LPIPS
    if normalize:
        transform_list.append(
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        )

    transform = transforms.Compose(transform_list)

    # Добавляем batch dimension
    tensor = transform(image).unsqueeze(0)

    return tensor


def load_image_batch(
    paths: list,
    preprocess_fn: callable = None
) -> list:
    """
    Загружает батч изображений.

    """
    images = []
    for path in paths:
        img = load_image(path)
        if preprocess_fn:
            img = preprocess_fn(img)
        images.append(img)
    return images


# Тестирование при запуске напрямую
if __name__ == "__main__":
    print("=" * 50)
    print("Тестирование утилит загрузки изображений")
    print("=" * 50)

    # Создаём тестовое изображение
    test_image = Image.new("RGB", (512, 512), color="red")
    test_path = "/tmp/test_image.png"
    test_image.save(test_path)

    print("\n1. Загрузка изображения:")
    img = load_image(test_path)
    print(f"   Размер: {img.size}")
    print(f"   Режим: {img.mode}")

    print("\n2. Предобработка для CLIP:")
    clip_tensor = preprocess_for_clip(test_path)
    print(f"   Shape: {clip_tensor.shape}")
    print(f"   Dtype: {clip_tensor.dtype}")
    print(
        f"   Min/Max: {clip_tensor.min().item()}, {clip_tensor.max().item()}")

    print("\n3. Предобработка для LPIPS:")
    lpips_tensor = preprocess_for_lpips(test_path)
    print(f"   Shape: {lpips_tensor.shape}")
    print(f"   Dtype: {lpips_tensor.dtype}")
    print(
        f"   Min/Max: {lpips_tensor.min().item():.2f}, {lpips_tensor.max().item():.2f}")

    print("\n✅ Все тесты пройдены!")
