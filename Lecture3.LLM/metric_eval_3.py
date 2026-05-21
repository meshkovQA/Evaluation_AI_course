# embedding_metrics_demo.py

from ai_client import openai_chat_v2
from typing import List, Dict
from embedding_metrics import (
    calculate_bertscore,
    calculate_semantic_similarity,
    calculate_coherence_sbert
)


def print_metric_per_text(metric_name: str, results: Dict, texts: List[str]):
    print(f"\n--- {metric_name.upper()} ---")

    if metric_name == "BERTScore":
        print(f"Общие показатели:")
        print(f"  Precision: {results['overall']['precision']:.3f}")
        print(f"  Recall: {results['overall']['recall']:.3f}")
        print(f"  F1: {results['overall']['f1']:.3f}")

        print(f"Попарные сравнения:")
        for i, (ref, cand, f1) in enumerate(results['individual']):
            print(f"  Пара {i+1}: F1 = {f1:.3f}")

    elif metric_name == "Semantic Similarity":
        print(
            f"Средняя семантическая близость: {results['overall_similarity_mean']:.3f}")
        print(f"Попарные сходства:")
        n_texts = len(texts)
        pair_idx = 0
        for i in range(n_texts):
            for j in range(i + 1, n_texts):
                similarity = results['pairwise'][pair_idx]
                print(f"  Текст {i+1} ↔ Текст {j+1}: {similarity:.3f}")
                pair_idx += 1

    elif metric_name == "Coherence SBERT":
        print(f"Общая когерентность: {results['overall']:.3f}")
        print(f"По текстам:")
        for i, score in enumerate(results['individual']):
            if score is not None:
                print(f"  Текст {i+1}: {score:.3f}")
            else:
                print(f"  Текст {i+1}: N/A (мало предложений)")

    print("-" * 50)


if __name__ == "__main__":
    print("=== ТЕСТИРОВАНИЕ EMBEDDING-МЕТРИК ===")

    # Промпты
    prompts = [
        "Напишите короткий текст об искусственном интеллекте",
        "Напишите текст об искусственном интеллекте",
        # "Объясните простыми словами, что такое нейронные сети"
    ]

    print("Генерируем тексты...")
    texts = [openai_chat_v2(prompt) for prompt in prompts]

    print("\nСгенерированные тексты:")
    for i, text in enumerate(texts):
        print(f"{i+1}. {text}\n")

    print("=" * 60)

    # BERTSCORE
    print("\nВычисляем BERTSCORE...")
    bertscore_results = calculate_bertscore(texts, lang="ru")
    print_metric_per_text("BERTScore", bertscore_results, texts)

    # SEMANTIC SIMILARITY
    print("\nВычисляем SEMANTIC SIMILARITY...")
    similarity_results = calculate_semantic_similarity(texts)
    print_metric_per_text("Semantic Similarity", similarity_results, texts)

    # COHERENCE SBERT
    print("\nВычисляем COHERENCE SBERT...")
    coherence_results = calculate_coherence_sbert(texts)
    print_metric_per_text("Coherence SBERT", coherence_results, texts)
