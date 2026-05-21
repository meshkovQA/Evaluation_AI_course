import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, top_k_accuracy_score
from scipy.stats import entropy as scipy_entropy
from torchmetrics.classification import MulticlassCalibrationError, MulticlassAccuracy
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# Настройка устройства для вычислений
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")

# ============================================================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ (CIFAR-10)
# ============================================================================


def load_cifar10_data(batch_size=128, val_split=0.1):
    """
    Загрузка и подготовка датасета CIFAR-10 с разделением на train/validation/test

    CIFAR-10 содержит 60,000 цветных изображений 32x32 пикселя в 10 классах:
    - airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

    Args:
        batch_size: размер батча
        val_split: доля валидационного набора от обучающего (0.1 = 10%)
    """

    # Определяем трансформации для обучающего набора (с аугментацией)
    transform_train = transforms.Compose([
        # RandomHorizontalFlip: случайно отражает картинку по горизонтали (например, самолёт влево/вправо).
        transforms.RandomHorizontalFlip(p=0.5),
        # RandomCrop: случайно вырезает часть картинки и добавляет рамку (чтобы имитировать разные ракурсы).
        transforms.RandomCrop(32, padding=4),
        # ToTensor: переводит картинку из формата PIL/NumPy в тензор PyTorch.
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))  # Нормализация
    ])

    # Трансформации для валидационного и тестового наборов (без аугментации)
    transform_val_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    # Загрузка исходных данных
    full_train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_val_test
    )

    # Разделение обучающего набора на train и validation
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    # Создание генератора для воспроизводимого разделения
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset_temp = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )

    # Создание отдельного датасета для валидации без аугментации
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False, transform=transform_val_test
    )

    # Получение индексов для валидационного набора
    val_indices = val_dataset_temp.indices
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    # Создание загрузчиков данных
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Названия классов CIFAR-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    print(f"Размеры наборов данных:")
    print(f"  - Обучающий: {len(train_dataset)} образцов")
    print(f"  - Валидационный: {len(val_dataset)} образцов")
    print(f"  - Тестовый: {len(test_dataset)} образцов")

    return train_loader, val_loader, test_loader, classes


# def load_mnist_data(batch_size=128, val_split=0.1):
#     """
#     Загрузка и подготовка датасета MNIST с разделением на train/validation/test

#     MNIST содержит 70,000 изображений рукописных цифр 28x28 пикселей в оттенках серого:
#     - цифры от 0 до 9

#     Args:
#         batch_size: размер батча
#         val_split: доля валидационного набора от обучающего (0.1 = 10%)
#     """

#     # Определяем трансформации для обучающего набора (с аугментацией)
#     transform_train = transforms.Compose([
#         # Случайное отражение по горизонтали (для цифр менее актуально)
#         transforms.RandomHorizontalFlip(p=0.1),  # Уменьшили вероятность
#         # Случайная обрезка с паддингом
#         transforms.RandomCrop(28, padding=2),     # Меньший padding для 28x28
#         # Случайный поворот (полезно для цифр)
#         transforms.RandomRotation(degrees=10),
#         transforms.ToTensor(),                    # Конвертация в тензор
#         #  нормализация для MNIST
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])

#     # Трансформации для валидационного и тестового наборов (без аугментации)
#     transform_val_test = transforms.Compose([
#         transforms.ToTensor(),
#         #  нормализация для MNIST
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])

#     # Загрузка MNIST вместо CIFAR-10
#     full_train_dataset = torchvision.datasets.MNIST(
#         root='./data', train=True, download=True, transform=transform_train
#     )
#     test_dataset = torchvision.datasets.MNIST(
#         root='./data', train=False, download=True, transform=transform_val_test
#     )

#     # Разделение обучающего набора на train и validation (БЕЗ ИЗМЕНЕНИЙ)
#     train_size = int((1 - val_split) * len(full_train_dataset))
#     val_size = len(full_train_dataset) - train_size

#     # Создание генератора для воспроизводимого разделения
#     generator = torch.Generator().manual_seed(42)
#     train_dataset, val_dataset_temp = torch.utils.data.random_split(
#         full_train_dataset, [train_size, val_size], generator=generator
#     )

#     # Создание отдельного датасета для валидации без аугментации
#     val_dataset = torchvision.datasets.MNIST(
#         root='./data', train=True, download=False, transform=transform_val_test
#     )

#     # Получение индексов для валидационного набора
#     val_indices = val_dataset_temp.indices
#     val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

#     # Создание загрузчиков данных (БЕЗ ИЗМЕНЕНИЙ)
#     train_loader = DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#     val_loader = DataLoader(
#         val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
#     test_loader = DataLoader(
#         test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

#     # Названия классов MNIST
#     classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

#     print(f"Размеры наборов данных:")
#     print(f"  - Обучающий: {len(train_dataset)} образцов")
#     print(f"  - Валидационный: {len(val_dataset)} образцов")
#     print(f"  - Тестовый: {len(test_dataset)} образцов")

#     return train_loader, val_loader, test_loader, classes

# ============================================================================
# 2. ОПРЕДЕЛЕНИЕ АРХИТЕКТУРЫ НЕЙРОННОЙ СЕТИ
# ============================================================================


class SimpleCNN_CIFAR(nn.Module):
    """
    Простая сверточная нейронная сеть для классификации CIFAR
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN_CIFAR, self).__init__()

        # Первый слой принимает 3 канала вместо 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)     # 32x32x3 -> 32x32x32
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)    # 32x32x32 -> 32x32x64
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)   # 16x16x64 -> 16x16x128

        # Пулинг и активация
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        #  Полносвязные слои адаптированы под размер 8x8x128
        # 8x8x128 -> 256 (меньше нейронов)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 64)           # 256 -> 64
        self.fc3 = nn.Linear(64, num_classes)   # 64 -> 10 (количество классов)

    def forward(self, x):
        # Сверточные слои с активацией и пулингом
        x = self.pool(self.relu(self.conv1(x)))  # 28x28x32 -> 14x14x32
        x = self.pool(self.relu(self.conv2(x)))  # 14x14x64 -> 7x7x64
        x = self.relu(self.conv3(x))             # 7x7x128 (БЕЗ пулинга)

        # Преобразование в одномерный вектор
        x = x.view(x.size(0), -1)  # Flatten: batch_size x (7*7*128)

        # Полносвязные слои
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # Выходной слой (логиты)

        return x


# class VerySimpleCNN(nn.Module):
#     """
#     Очень простая модель для сравнения
#     """

#     def __init__(self, num_classes=10):
#         super(VerySimpleCNN, self).__init__()

#         self.conv1 = nn.Conv2d(1, 16, 5)  # 28x28 -> 24x24
#         self.pool = nn.MaxPool2d(2, 2)    # 24x24 -> 12x12
#         self.conv2 = nn.Conv2d(16, 32, 5)  # 12x12 -> 8x8
#         self.fc1 = nn.Linear(32 * 4 * 4, 64)
#         self.fc2 = nn.Linear(64, num_classes)

#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = x.view(-1, 32 * 4 * 4)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


# class SimpleCNN(nn.Module):
#     """
#     Простая сверточная нейронная сеть для классификации изображений CIFAR-10
#     """

#     def __init__(self, num_classes=10):
#         super(SimpleCNN, self).__init__()
#         # 1) входных каналов = 1
#         self.conv1 = nn.Conv2d(1, 32, 3, padding=1)     # 28x28 -> 28x28
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)    # 14x14 -> 14x14
#         self.conv3 = nn.Conv2d(64, 128, 3, padding=1)   # 7x7   -> 7x7
#         # 3x3   -> 3x3 (после 3 пулов)
#         self.conv4 = nn.Conv2d(128, 256, 3, padding=1)

#         self.pool = nn.MaxPool2d(2, 2)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)

#         # 2) размер после трех пулов на 28x28 = 3x3
#         self.fc1 = nn.Linear(256 * 3 * 3, 512)
#         self.fc2 = nn.Linear(512, 128)
#         self.fc3 = nn.Linear(128, num_classes)

#     def forward(self, x):
#         # Сверточные слои с активацией и пулингом
#         x = self.pool(self.relu(self.conv1(x)))  # 32x32x32 -> 16x16x32
#         x = self.pool(self.relu(self.conv2(x)))  # 16x16x64 -> 8x8x64
#         x = self.pool(self.relu(self.conv3(x)))  # 8x8x128 -> 4x4x128
#         x = self.relu(self.conv4(x))             # 4x4x256

#         # Преобразование в одномерный вектор
#         x = x.view(x.size(0), -1)  # Flatten: batch_size x (4*4*256)

#         # Полносвязные слои
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.fc3(x)  # Выходной слой (логиты)

#         return x

# ============================================================================
# 3. ФУНКЦИИ ОБУЧЕНИЯ И ОЦЕНКИ
# ============================================================================


def validate_model(model, val_loader, criterion, device):
    """
    Валидация модели (оценка на валидационном наборе)
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    return avg_val_loss, val_accuracy


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    """
    Обучение модели с валидацией и отслеживанием метрик
    """
    # Списки для хранения метрик
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Для раннего останова (early stopping)
    best_val_accuracy = 0.0
    patience = 3  # Количество эпох без улучшения
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # ====================================================================
        # ОБУЧЕНИЕ
        # ====================================================================
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            # Перемещение данных на устройство
            inputs, labels = inputs.to(device), labels.to(device)

            # Обнуление градиентов
            optimizer.zero_grad()

            # Прямой проход
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Обратный проход и оптимизация
            loss.backward()
            optimizer.step()

            # Статистика
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Метрики обучения за эпоху
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # ====================================================================
        # ВАЛИДАЦИЯ
        # ====================================================================
        val_loss, val_accuracy = validate_model(
            model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Печать метрик эпохи
        print(f'Эпоха [{epoch+1}/{num_epochs}]:')
        print(
            f'  Обучение  - Потеря: {train_loss:.4f}, Точность: {train_accuracy:.2f}%')
        print(
            f'  Валидация - Потеря: {val_loss:.4f}, Точность: {val_accuracy:.2f}%')
        print('-' * 60)

        # Проверка на улучшение (для early stopping)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            # Здесь можно сохранить лучшую модель
            print(
                f'  ✅ Новая лучшая валидационная точность: {best_val_accuracy:.2f}%')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f'  ⏹️ Ранний останов: {patience} эпох без улучшения')
                break

        print()

    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_accuracy': best_val_accuracy
    }


def evaluate_model_with_metrics(model, test_loader, device, classes):
    """
    Комплексная оценка модели с различными метриками
    """
    model.eval()

    # Списки для хранения результатов
    all_predictions = []
    all_labels = []
    all_softmax_probs = []
    all_logits = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Получение логитов (сырых выходов)
            logits = model(inputs)

            # Применение softmax для получения вероятностей
            softmax_probs = torch.softmax(logits, dim=1)

            # Получение предсказаний
            _, predicted = torch.max(logits, 1)

            # Сохранение результатов
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_softmax_probs.extend(softmax_probs.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())

    # Преобразование в numpy массивы для удобства
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_softmax_probs = np.array(all_softmax_probs)
    all_logits = np.array(all_logits)

    return all_predictions, all_labels, all_softmax_probs, all_logits

# ============================================================================
# 4. ФУНКЦИИ ДЛЯ ВЫЧИСЛЕНИЯ МЕТРИК
# ============================================================================


def calculate_confidence_scores(softmax_probs):
    """
    Вычисление confidence scores как максимальной вероятности
    """
    confidence_scores = np.max(softmax_probs, axis=1)
    return confidence_scores


def calculate_top_k_accuracy(softmax_probs, true_labels, k=5):
    """
    Вычисление Top-K accuracy
    """
    acc = top_k_accuracy_score(true_labels, softmax_probs, k=k)
    return acc * 100.0


def calculate_entropy(softmax_probs):
    """
    Вычисление энтропии для измерения неопределенности
    H = -∑ p_i * log(p_i)
    """
    # Добавление малого значения для избежания log(0)
    ent = scipy_entropy(softmax_probs, axis=1)
    return ent


def calculate_calibration_error(softmax_probs, predictions, true_labels, n_bins=10):
    """
    Вычисление Expected Calibration Error (ECE)
    Измеряет соответствие между уверенностью модели и реальной точностью
    """
    probs_t = torch.from_numpy(softmax_probs).float()      # shape: [N, C]
    targets_t = torch.from_numpy(true_labels).long()       # shape: [N]

    # ECE с L1-нормой (|conf - acc|), как у тебя было
    ece_metric = MulticlassCalibrationError(
        num_classes=softmax_probs.shape[1],
        n_bins=n_bins,
        norm='l1'
    )
    ece = float(ece_metric(probs_t, targets_t).item())

    # bin_data не нужен — возвращаем пустой список, чтобы не менять сигнатуры
    return ece, []

# ============================================================================
# 5. ФУНКЦИИ ВИЗУАЛИЗАЦИИ
# ============================================================================


def plot_training_history(training_history):
    """
    Построение графиков обучения и валидации (с лентами диапазона)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # --- единый стиль
    FIGSIZE = (12, 5)
    GRID_ALPHA = 0.3
    BAND_ALPHA = 0.15
    LINEWIDTH = 2

    train_accuracies = np.asarray(
        training_history['train_accuracies'], dtype=float)
    val_accuracies = np.asarray(
        training_history['val_accuracies'],   dtype=float)
    train_losses = np.asarray(
        training_history['train_losses'],     dtype=float)
    val_losses = np.asarray(training_history['val_losses'],       dtype=float)
    epochs = np.arange(1, len(train_accuracies) + 1)

    def _smooth(x, k=3):
        if len(x) < k:
            return x
        w = np.ones(k)/k
        y = np.convolve(x, w, mode='valid')
        pad = (len(x) - len(y)) // 2
        return np.pad(y, (pad, len(x)-len(y)-pad), mode='edge')

    def _roll_minmax(x, k=5):
        if k < 2 or len(x) < k:
            return x, x
        from collections import deque
        dmin, dmax, qmin, qmax, xs = [], [], deque(), deque(), x.tolist()
        for i, v in enumerate(xs):
            while qmin and xs[qmin[-1]] >= v:
                qmin.pop()
            qmin.append(i)
            while qmax and xs[qmax[-1]] <= v:
                qmax.pop()
            qmax.append(i)
            if qmin[0] <= i-k:
                qmin.popleft()
            if qmax[0] <= i-k:
                qmax.popleft()
            dmin.append(xs[qmin[0]])
            dmax.append(xs[qmax[0]])
        return np.array(dmin), np.array(dmax)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)

    # --- Точность
    tr_s = _smooth(train_accuracies, k=3)
    va_s = _smooth(val_accuracies,   k=3)
    ax1.plot(epochs, tr_s, linewidth=LINEWIDTH, label='Обучение')
    ax1.plot(epochs, va_s, linewidth=LINEWIDTH, label='Валидация')
    tr_lo, tr_hi = _roll_minmax(train_accuracies, k=5)
    va_lo, va_hi = _roll_minmax(val_accuracies,   k=5)
    ax1.fill_between(epochs, tr_lo, tr_hi, alpha=BAND_ALPHA)
    ax1.fill_between(epochs, va_lo, va_hi, alpha=BAND_ALPHA)
    ax1.set_title('Точность по эпохам')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Точность (%)')
    ax1.grid(True, alpha=GRID_ALPHA)
    ax1.legend(loc='lower right')

    # --- Потери
    tl_s = _smooth(train_losses, k=3)
    vl_s = _smooth(val_losses,   k=3)
    ax2.plot(epochs, tl_s, linewidth=LINEWIDTH, label='Обучение')
    ax2.plot(epochs, vl_s, linewidth=LINEWIDTH, label='Валидация')
    tl_lo, tl_hi = _roll_minmax(train_losses, k=5)
    vl_lo, vl_hi = _roll_minmax(val_losses,   k=5)
    ax2.fill_between(epochs, tl_lo, tl_hi, alpha=BAND_ALPHA)
    ax2.fill_between(epochs, vl_lo, vl_hi, alpha=BAND_ALPHA)
    ax2.set_title('Потери по эпохам')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Потеря')
    ax2.grid(True, alpha=GRID_ALPHA)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    # --- Анализ переобучения (без изменений)
    print("\n🔍 АНАЛИЗ ПЕРЕОБУЧЕНИЯ:")
    final_train_acc = float(train_accuracies[-1])
    final_val_acc = float(val_accuracies[-1])
    gap = final_train_acc - final_val_acc
    print(f"Финальная точность обучения: {final_train_acc:.2f}%")
    print(f"Финальная точность валидации: {final_val_acc:.2f}%")
    print(f"Разрыв (Train - Val): {gap:.2f}%")
    if gap < 3:
        print("✅ Переобучения практически нет")
    elif gap < 8:
        print("⚠️ Легкое переобучение")
    elif gap < 15:
        print("🔶 Умеренное переобучение")
    else:
        print("❌ Сильное переобучение")
    return gap


def plot_confusion_matrix(true_labels, predictions, classes):
    """
    Построение матрицы ошибок (проценты + абсолюты, нормировка по истине)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    FIGSIZE = (8, 6)
    GRID_ALPHA = 0.3

    cm_abs = confusion_matrix(true_labels, predictions)
    with np.errstate(invalid='ignore'):
        cm = cm_abs / cm_abs.sum(axis=1, keepdims=True)
    cm = np.nan_to_num(cm)

    plt.figure(figsize=FIGSIZE)
    ax = sns.heatmap(cm, annot=False, cmap='Blues',
                     xticklabels=classes, yticklabels=classes,
                     vmin=0.0, vmax=1.0, cbar_kws={"label": "Доля по истинному классу"})
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]*100:.1f}%\n({cm_abs[i, j]})"
            ax.text(j+0.5, i+0.5, txt, ha='center',
                    va='center', fontsize=9, color='black')

    plt.title('Матрица путаницы (ошибок)')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.grid(False)  # у heatmap своя сетка
    plt.tight_layout()
    plt.show()


# ============================================================================
# 6. ГЛАВНАЯ ФУНКЦИЯ - ПОЛНЫЙ ПАЙПЛАЙН
# ============================================================================


def main():
    """
    Основная функция, демонстрирующая полный пайплайн обучения и оценки
    """
    print("=" * 80)
    print("ПРАКТИКА: КЛАССИФИКАЦИЯ И МЕТРИКИ ГЛУБОКОГО ОБУЧЕНИЯ")
    print("=" * 80)

    # 1. Загрузка данных
    print("\n1. Загрузка данных CIFAR-10...")
    train_loader, val_loader, test_loader, classes = load_cifar10_data(
        batch_size=128, val_split=0.1)
    print(f"Классы: {classes}")

    # 2. Создание модели
    print("\n2. Создание модели...")
    model = SimpleCNN_CIFAR(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Подсчет параметров
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Всего параметров в модели: {total_params:,}")

    # 3. Обучение модели
    print("\n3. Обучение модели с валидацией...")
    print("=" * 60)
    training_history = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10
    )

    print(f"\n🎯 ИТОГИ ОБУЧЕНИЯ:")
    print(
        f"Лучшая валидационная точность: {training_history['best_val_accuracy']:.2f}%")

    # 4. Финальная оценка на тестовом наборе
    print("\n4. Финальная оценка на тестовом наборе...")
    print("⚠️  ВАЖНО: Тестовый набор используется ТОЛЬКО для финальной оценки!")
    print("🔒 Этот набор НЕ использовался во время обучения и валидации")
    predictions, true_labels, softmax_probs, logits = evaluate_model_with_metrics(
        model, test_loader, device, classes
    )

    # Базовая точность на тестовом наборе
    test_accuracy = accuracy_score(true_labels, predictions) * 100
    val_accuracy = training_history['best_val_accuracy']

    print(f"\n📊 СРАВНЕНИЕ ВАЛИДАЦИИ И ТЕСТИРОВАНИЯ:")
    print(f"Лучшая валидационная точность: {val_accuracy:.2f}%")
    print(f"Финальная тестовая точность:   {test_accuracy:.2f}%")

    accuracy_diff = abs(val_accuracy - test_accuracy)
    if accuracy_diff < 2:
        print(f"✅ Отличное соответствие (разница: {accuracy_diff:.2f}%)")
    elif accuracy_diff < 5:
        print(f"✅ Хорошее соответствие (разница: {accuracy_diff:.2f}%)")
    elif accuracy_diff < 10:
        print(f"⚠️  Умеренное расхождение (разница: {accuracy_diff:.2f}%)")
    else:
        print(f"❌ Значительное расхождение (разница: {accuracy_diff:.2f}%)")
        print("   Возможные причины: переобучение на валидации, различия в данных")

    # ========================================================================
    # 5. ВЫЧИСЛЕНИЕ И АНАЛИЗ МЕТРИК
    # ========================================================================

    print("\n" + "=" * 50)
    print("ФИНАЛЬНЫЕ МЕТРИКИ НА ТЕСТОВОМ НАБОРЕ")
    print("=" * 50)
    print("🔬 Все нижеприведенные метрики вычислены на независимом тест сете")

    # 5.1 Базовая точность (Top-1 Accuracy)
    basic_accuracy = accuracy_score(true_labels, predictions) * 100
    print(f"\n📊 БАЗОВЫЕ МЕТРИКИ:")
    print(f"Top-1 Accuracy: {basic_accuracy:.2f}%")

    # 5.2 Confidence Scores
    confidence_scores = calculate_confidence_scores(softmax_probs)
    avg_confidence = np.mean(confidence_scores)
    print(f"Средний Confidence Score: {avg_confidence:.3f}")
    print(f"Мин. Confidence Score: {np.min(confidence_scores):.3f}")
    print(f"Макс. Confidence Score: {np.max(confidence_scores):.3f}")

    # 5.3 Top-K Accuracy
    print(f"\n🎯 TOP-K ACCURACY:")
    for k in [1, 3, 5]:
        top_k_acc = calculate_top_k_accuracy(softmax_probs, true_labels, k=k)
        print(f"Top-{k} Accuracy: {top_k_acc:.2f}%")

    # 5.4 Энтропия
    entropy_values = calculate_entropy(softmax_probs)
    avg_entropy = np.mean(entropy_values)
    print(f"\n🔀 ЭНТРОПИЯ (неопределенность):")
    print(f"Средняя энтропия: {avg_entropy:.3f}")
    print(f"Мин. энтропия: {np.min(entropy_values):.3f}")
    print(f"Макс. энтропия: {np.max(entropy_values):.3f}")
    print(f"Стандартное отклонение: {np.std(entropy_values):.3f}")

    # 5.5 Калибровка
    ece, bin_data = calculate_calibration_error(
        softmax_probs, predictions, true_labels)
    print(f"\n⚖️ КАЛИБРОВКА:")
    print(
        f"Expected Calibration Error (ECE) Разница между заявленной уверенностью и реальной точностью всего: {ece:.3f}")
    print(f"Как интерпритировать: если модель говорит «я уверена на 80%», то она примерно в 76–80% случаев права (почти идеально).")
    print("Интерпретация ECE:")
    print("  - 0.0-0.05: Отлично откалибрована")
    print("  - 0.05-0.1: Хорошо откалибрована")
    print("  - 0.1-0.2: Умеренно откалибрована")
    print("  - >0.2: Плохо откалибрована")

    # 5.6 Анализ по классам
    print(f"\n📋 ДЕТАЛЬНЫЙ ОТЧЕТ ПО КЛАССАМ:")
    class_report = classification_report(true_labels, predictions,
                                         target_names=classes, digits=3)
    print(class_report)

    # ========================================================================
    # 6. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
    # ========================================================================

    print("\n" + "=" * 50)
    print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("=" * 50)

    # График обучения с валидацией
    overfitting_gap = plot_training_history(training_history)

    # Матрица ошибок
    plot_confusion_matrix(true_labels, predictions, classes)

# ============================================================================
# 8. ЗАПУСК ПРОГРАММЫ
# ============================================================================


if __name__ == "__main__":
    main()
