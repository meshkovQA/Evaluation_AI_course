# metric_eval_2.py

from reference_free import (
    calculate_distinct,
    calculate_perplexity,
    calculate_readability,
    calculate_self_bleu,
    calculate_coherence_with_transformers,
    calculate_grammar_errors
)
from typing import List, Dict
from ai_client import openai_chat_v2

"""
    Как читать метрики:

    PERPLEXITY (перплексия) - Перплексия показывает, насколько “естественным” и грамматически нормальным выглядит текст с точки зрения языковой модели.
    - Чем меньше значение, тем более естественный и «понятный» текст.
    - Высокие значения — значит, что слова кажутся модели непривычными или структура текста странная.

    READABILITY (читаемость, индекс Флеша). Метрика оценивает, насколько легко человеку прочитать текст.
    Значения обычно варьируются от 0 до 100:
        •	90–100 — очень легко читается (детские тексты)
        •	60–70 — обычный уровень (новости, статьи)
        •	30–50 — сложные тексты (академические, технические)
        •	<30 — очень трудные для чтения

    GRAMMAR ERRORS (грамматические ошибки). Количество грамматических ошибок, найденных моделью или инструментом (часто LanguageTool / GPT детектор).
    - Чем меньше показатель, тем лучше. 0 — идеально, без ошибок.

    СOHERENCE (когерентность). Насколько логично и связно построен текст.
    - Значения от 0 до 1, где 1 — идеально связный текст
    Обычно:
	•	0.6–0.7 — средняя связность
	•	0.8+ — отличная логика и плавность

    SELF-BLEU (само-BLEU). Насколько разнообразен текст по сравнению с самим собой.
    - Значения от 0 до 1, где 0 — максимально разнообразный текст, 1 — полностью повторяющийся.
    Обычно:
    •	0.3–0.5 — хороший уровень разнообразия  
    •	0.6+ — текст может быть слишком повторяющимся

    DISTINCT (разнообразие). Насколько разнообразен текст по количеству уникальных n-грамм.
    Показывает долю уникальных слов и фраз:
	•	distinct_1 — уникальных слов (униграмм)
	•	distinct_2 — уникальных пар слов (биграмм)

    DISTINCT-1 — разнообразие на уровне слов (униграмм)
    Чем выше — тем разнообразнее и “живее” текст.
    - Низкие значения 0.1–0.3 — текст может быть монотонным и повторяющимся
    - Средние значения 0.3–0.6 — текст сбалансирован по разнообразию. Нормально для новостных, технических, FAQ-текстов
    - Высокие значения 0.6-0.8 — текст разнообразный, естественная вариативность
    - Очень высокие значения 0.8+ — часто в креативных или длинных текстах

    DISTINCT-2 — разнообразие на уровне фраз (биграмм)
    Чем выше — тем более разнообразные фразы и конструкции использованы.
    - Низкие значения 0.2–0.8 — повторяются одинаковые связки («в этой статье», «можно использовать»)
    - Средние значения 0.8–1.5 — нормальное разнообразие фраз. Приемлемо для новостей и научных статей
    - Высокие значения 1.5-2.5 — текст с хорошим разнообразием фраз и конструкций, естественный язык
    - Очень высокие значения 2.5+ — очень вариативно, креативный стиль 


    """


def print_metric_per_text(metric_name: str, results: List, texts: List[str]):
    print(f"\n--- {metric_name.upper()} ---")
    for i, (text, score) in enumerate(zip(texts, results)):
        print(f"Текст {i+1}: {text[:50]}...")
        if isinstance(score, dict):
            for key, val in score.items():
                print(f"  {key}: {val:.3f}")
        else:
            print(f"  Скор: {score:.3f}")
    print("-" * 50)


if __name__ == "__main__":
    print("=== ТЕСТИРОВАНИЕ REFERENCE-FREE МЕТРИК ===")

    # Промпты
    prompts = [
        "Напишите короткий текст об искусственном интеллекте",
        "Опишите преимущества машинного обучения",
        "Объясните простыми словами, что такое нейронные сети"
    ]

    print("Генерируем тексты...")
    texts = [openai_chat_v2(prompt) for prompt in prompts]

    print("\nСгенерированные тексты:")
    for i, text in enumerate(texts):
        print(f"{i+1}. {text}\n")

    print("=" * 60)

    # PERPLEXITY
    print("\nВычисляем PERPLEXITY...")
    perplexities = [calculate_perplexity(text) for text in texts]
    print_metric_per_text("Perplexity", perplexities, texts)

    # READABILITY
    print("\nВычисляем READABILITY...")
    readability_scores = [calculate_readability(text) for text in texts]
    print_metric_per_text("Readability", readability_scores, texts)

    # GRAMMAR ERRORS
    print("\nВычисляем GRAMMAR ERRORS...")
    grammar_errors = [calculate_grammar_errors(text) for text in texts]
    print_metric_per_text("Grammar Errors", grammar_errors, texts)

    # COHERENCE
    print("\nВычисляем COHERENCE...")
    coherence_scores = [
        calculate_coherence_with_transformers(text) for text in texts]
    print_metric_per_text("Coherence", coherence_scores, texts)

    # SELF-BLEU
    print("\nВычисляем SELF-BLEU...")
    self_bleu_scores = [calculate_self_bleu(
        i, texts, n=2) for i in range(len(texts))]
    print_metric_per_text("Self-BLEU (2-gram)", self_bleu_scores, texts)

    # DISTINCT
    print("\nВычисляем DISTINCT...")
    distinct_result = calculate_distinct(texts)
    print(f"\n--- DISTINCT ---")
    for key, val in distinct_result.items():
        print(f"{key}: {val:.3f}")
    print("-" * 50)
