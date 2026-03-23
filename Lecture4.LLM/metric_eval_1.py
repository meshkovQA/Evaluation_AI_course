from ai_client import openai_chat_v2
from typing import List, Dict, Any
from reference_based import calculate_rouge, calculate_meteor, calculate_sacrebleu, calculate_exact_match


def print_detailed_results(metric_name: str, results: Dict, prompts: List[str], predictions: List[str], references: List[str]):
    """Печать детальных результатов метрики"""
    print(f"\n--- {metric_name} ---")
    if isinstance(results["overall"], dict):
        # Для ROUGE
        for key, value in results["overall"].items():
            print(f"Общий {key.upper()}: {value:.3f}")
        print("\nПо каждому запросу:")
        for i, (prompt, pred, ref) in enumerate(zip(prompts, predictions, references)):
            print(f"Запрос:")
            print(f"  {prompt}")
            print(f"Ответ от LLM:")
            print(f"  {pred}")
            print(f"Референс:")
            print(f"  {ref}")
            for key in results["individual"]:
                print(f"  {key.upper()}: {results['individual'][key][i]:.3f}")
            print()
    else:
        # Для остальных метрик
        print(f"Общий: {results['overall']:.3f}")
        print("\nПо каждому запросу:")
        for i, (prompt, pred, ref, score) in enumerate(zip(prompts, predictions, references, results["individual"])):
            print(f"Запрос:")
            print(f"  {prompt}")
            print(f"Ответ от LLM:")
            print(f"  {pred}")
            print(f"Референс:")
            print(f"  {ref}")
            print(f"  Скор: {score:.3f}")
            print()


if __name__ == "__main__":
    # ---------  Пример тестирования перевода ----------

    print("=== ТЕСТИРОВАНИЕ ПЕРЕВОДА ===")

    translation_prompts = [
        "Переведи на русский: Good morning",
        "Переведи на русский: Beautiful weather",
        "Переведи на русский: I love books"
    ]

    translation_references = [
        "Доброе утро",
        "Прекрасная погода",
        "Я люблю книги"
    ]

    print("Генерируем переводы...")
    translation_predictions = [openai_chat_v2(
        prompt) for prompt in translation_prompts]

    print("Сгенерированные переводы:")
    for i, pred in enumerate(translation_predictions):
        print(f"  {i+1}. {pred}")

    meteor_results = calculate_meteor(
        translation_predictions, translation_references)

    print_detailed_results("METEOR", meteor_results, translation_prompts,
                           translation_predictions, translation_references)

    # ---------  Пример тестирования суммаризации ----------

    print("\n\n=== ТЕСТИРОВАНИЕ СУММАРИЗАЦИИ ===")

    summarization_prompts = [
        "Summarize in one sentence: Artificial intelligence is rapidly transforming industries through machine learning algorithms that can process vast amounts of data and identify patterns that humans might miss.",
        "Summarize in one sentence: Climate change represents one of the most pressing challenges of our time, requiring immediate global action to reduce greenhouse gas emissions."
    ]

    summarization_references = [
        "artificial intelligence transforms industries through machine learning algorithms for data analysis and pattern detection",
        "climate change requires urgent global measures to reduce greenhouse gas emissions"
    ]

    print("Генерируем суммаризации...")
    summarization_predictions = [openai_chat_v2(
        prompt) for prompt in summarization_prompts]

    print("Сгенерированные суммаризации:")
    for i, pred in enumerate(summarization_predictions):
        print(f"  {i+1}. {pred}")

    rouge_results = calculate_rouge(
        summarization_predictions, summarization_references)
    print_detailed_results("ROUGE", rouge_results, summarization_prompts,
                           summarization_predictions, summarization_references)

    # ---------  Пример тестирования точного перевода ----------

    print("\n\n=== ТЕСТИРОВАНИЕ ТОЧНОГО ПЕРЕВОДА ===")

    precise_translation_prompts = [
        "Переведи точно на английский: Привет мир",
        "Переведи точно на английский: Спасибо"
    ]

    precise_translation_references = [
        "Hello world",
        "Thank you"
    ]

    print("Генерируем точные переводы...")
    precise_predictions = [openai_chat_v2(
        prompt) for prompt in precise_translation_prompts]

    print("Сгенерированные переводы:")
    for i, pred in enumerate(precise_predictions):
        print(f"  {i+1}. {pred}")

    sacrebleu_results = calculate_sacrebleu(
        precise_predictions, precise_translation_references)
    print_detailed_results("SacreBLEU", sacrebleu_results, precise_translation_prompts,
                           precise_predictions, precise_translation_references)

    # ---------  Пример тестирования точности ответов (QA) ----------

    print("\n\n=== ТЕСТИРОВАНИЕ ТОЧНЫХ ОТВЕТОВ (QA) ===")

    qa_prompts = [
        "Какая столица Франции? Ответь одним словом.",
        "Сколько будет 2+2? Ответь одной цифрой.",
        "Какой цвет получается при смешивании красного и синего? Ответь одним словом."
    ]

    qa_references = [
        "Париж",
        "4",
        "Фиолетовый"
    ]

    print("Генерируем ответы...")
    qa_predictions = [openai_chat_v2(prompt) for prompt in qa_prompts]

    print("Сгенерированные ответы:")
    for i, pred in enumerate(qa_predictions):
        print(f"  {i+1}. {pred}")

    exact_match_results = calculate_exact_match(qa_predictions, qa_references)
    print_detailed_results("Exact Match", exact_match_results,
                           qa_prompts, qa_predictions, qa_references)
