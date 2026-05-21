# tensorboard_classification_demo.py
# =============================================================================
# КЛАССИФИКАЦИЯ ДЛЯ ГЛУБОКОГО ОБУЧЕНИЯ - ДАТАСЕТ WINE (СТАНДАРТНЫЙ TENSORBOARD)
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DataLoader:
    """Загрузка и предобработка данных"""

    def __init__(self):
        self.data = None
        self.scaler_X = StandardScaler()
        self.class_names = None

    def load_wine_dataset(self):
        print("\n🍷 Загружаем датасет Wine...")
        wine = load_wine()
        X, y = wine.data, wine.target
        feature_names = wine.feature_names
        self.class_names = wine.target_names

        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y
        self.data = df

        print(f"📋 Размер: {self.data.shape}")
        print(f"📋 Классы: {', '.join(self.class_names)}")
        class_counts = pd.Series(y).value_counts().sort_index()
        print("📊 Распределение классов:")
        for idx, cnt in class_counts.items():
            print(f"   • {self.class_names[idx]}: {cnt}")
        return df

    def prepare_data(self, test_size=0.2, val_size=0.25, random_state=42):
        print("\n🔧 Подготовка данных...")
        X = self.data.drop("target", axis=1).values
        y = self.data["target"].values

        X_scaled = self.scaler_X.fit_transform(X)

        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X_scaled, y, test_size=test_size, stratify=y, random_state=random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=val_size, stratify=y_tmp, random_state=random_state
        )

        print(
            f"📊 Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test


class ModelBuilder:
    @staticmethod
    def create_wine_classifier(input_dim, num_classes):
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.histories = {}

    def setup_callbacks(self, log_dir="logs/wine_quality_standard"):
        """Стандартные коллбэки TensorBoard + ранняя остановка/уменьшение lr"""
        tb = keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,     # вкладки Histograms/Distributions
            write_graph=True,
            update_freq='epoch'
        )
        early = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        )
        reduce = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        )
        return [tb, early, reduce]

    def train_model(self, model, model_name,
                    X_train, y_train, X_val, y_val,
                    epochs=30, batch_size=32, log_dir="logs/wine_quality_standard"):
        print(f"\n🚀 Обучение модели {model_name}...")

        # Веса классов (на всякий случай при дисбалансе)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        print(f"📊 Веса классов: {class_weight_dict}")

        callbacks = self.setup_callbacks(log_dir)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        self.models[model_name] = model
        self.histories[model_name] = history
        print(f"✅ Обучение {model_name} завершено")
        print(f"🔎 Открой стандартные графики TensorBoard:")
        print(
            f"   tensorboard --logdir {log_dir}\n   затем перейди в http://localhost:6006")
        return history


class Runner:
    def __init__(self):
        self.data = DataLoader()
        self.builder = ModelBuilder()
        self.trainer = ModelTrainer()

    def run(self):
        print("="*70)
        print("🍷 КЛАССИФИКАЦИЯ ВИНА — СТАНДАРТНЫЙ TENSORBOARD")
        print("="*70)

        self.data.load_wine_dataset()
        X_train, X_val, X_test, y_train, y_val, y_test = self.data.prepare_data()

        log_dir = "logs/wine_quality_standard"

        model = self.builder.create_wine_classifier(
            input_dim=X_train.shape[1],
            num_classes=len(np.unique(y_train))
        )

        print("\n🏗️ Архитектура модели:")
        model.summary()

        self.trainer.train_model(
            model, "Wine_Classifier",
            X_train, y_train, X_val, y_val,
            epochs=30, batch_size=32, log_dir=log_dir
        )

        print("\n📊 Оценка на тесте:")
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"   Loss: {test_loss:.4f}")
        print(f"   Accuracy: {test_acc:.4f}")


# ============================================================================
# ЗАПУСК ПРОГРАММЫ
# ============================================================================
def main():
    Runner().run()
    print("\n" + "="*80)
    print("🎉 Анализ завершён. Запусти TensorBoard и смотри стандартные графики!")
    print("="*80)


if __name__ == "__main__":
    main()
