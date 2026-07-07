"""
Создание датасета в DeepEval из Excel и ответов RAG
Использует готовые DatasetParser и RAGConnector
"""
import sys
from pathlib import Path
import time

# СНАЧАЛА добавляем путь
current_dir = Path(__file__).resolve().parent  # .../deepeval_lib
eval_rag_dir = current_dir.parent  # .../eval_rag
sys.path.insert(0, str(eval_rag_dir))

from deepeval.dataset import EvaluationDataset, Golden  # noqa: E402
from rag_connector import RAGConnector  # noqa: E402
from dataset_parser import DatasetParser  # noqa: E402


def create_deepeval_dataset_from_excel(
    excel_path: str,
    dataset_alias: str,
    rag_connector: RAGConnector,
    sleep_time: float = 0.1
):

    # ============= ШАГ 1: Парсим Excel =============

    parser = DatasetParser()
    df = parser.load_dataset(excel_path)

    if df is None:
        print("❌ Ошибка загрузки датасета")
        return

    # Показываем информацию
    info = parser.validate_dataset(df)
    print(f"\n📊 Информация о датасете:")
    print(f"   Всего строк: {info['total_rows']}")
    print(f"   Валидных пар: {info['valid_pairs']}")

    # ============= ШАГ 2: Получаем вопросы и expected ответы =============
    questions = parser.get_questions(df)
    expected_responses = parser.get_expected_responses(df)

    print(f"✅ Извлечено {len(questions)} вопросов")

    # ============= ШАГ 3: Задаем вопросы в RAG =============
    print(f"\n🤖 Шаг 3: Получение ответов от RAG системы...")

    goldens = []

    for i, question in enumerate(questions, 1):
        print(f"\n  [{i}/{len(questions)}] {question[:60]}...")

        # Запрос к RAG
        rag_response = rag_connector.query(question)

        # Проверяем на ошибки
        if 'error' in rag_response:
            print(f"      ⚠️  Ошибка RAG: {rag_response['error']}")
            continue

        # Извлекаем данные
        actual_output = rag_response.get('content', '')
        sources = rag_response.get('sources', [])
        retrieval_context = [s.get('content', '')
                             for s in sources if s.get('content')]

        print(f"      ✅ Ответ: {actual_output[:60]}...")
        print(f"      📚 Контекстов: {len(retrieval_context)}")

        # В deepeval 4.x Golden поддерживает actual_output и retrieval_context —
        # можно сохранить ответ RAG и его контексты прямо в датасете.
        golden = Golden(
            input=question,
            actual_output=actual_output,
            expected_output=expected_responses[i-1] if i -
            1 < len(expected_responses) else "",
            retrieval_context=retrieval_context,
        )

        goldens.append(golden)

        # Пауза между запросами
        time.sleep(sleep_time)

    # ============= ШАГ 4: Сохраняем в DeepEval =============
    print(f"\n💾 Шаг 4: Сохранение датасета в DeepEval...")

    try:
        # deepeval 4.x: EvaluationDataset принимает только goldens=,
        # а push() больше не имеет auto_convert_test_cases_to_goldens.
        evaluation_dataset = EvaluationDataset(goldens=goldens)
        evaluation_dataset.push(alias=dataset_alias)

        print(f"✅ Датасет '{dataset_alias}' успешно создан в DeepEval!")
        print(f"📊 Сохранено {len(goldens)} записей")
        print(f"🌐 Просмотрите датасет на https://app.confident-ai.com")

    except Exception as e:
        print(f"❌ Ошибка при сохранении в DeepEval: {e}")
        print("💡 Убедитесь что:")
        print("   1. Выполнили: deepeval login")
        print("   2. У вас есть интернет-соединение")
        print("   3. API ключ DeepEval валиден")


if __name__ == "__main__":

    # ============= НАСТРОЙКА =============

    # 1. Инициализируем RAG коннектор
    rag_connector = RAGConnector(
        endpoint_url="http://5.11.83.110:8002/api/v1/chat/",
        api_key="YOUR_API_KEY",  # ← Ваш API ключ
        timeout=30
    )

    # 2. Указываем путь к Excel файлу
    excel_path = "eval_rag/data/evaluation_dataset.xlsx"  # ← Ваш путь

    # 3. Алиас для датасета в DeepEval UI
    dataset_alias = "Second dataset"  # ← Название в UI

    # ============= ЗАПУСК =============

    create_deepeval_dataset_from_excel(
        excel_path=excel_path,
        dataset_alias=dataset_alias,
        rag_connector=rag_connector,
        sleep_time=0.1  # Пауза между запросами к RAG
    )
