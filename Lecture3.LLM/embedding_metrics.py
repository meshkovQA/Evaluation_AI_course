# embedding_metrics.py

"""
Метрики на основе эмбеддингов:
- BERTScore — сравнение текстов по токенам BERT
- Sentence-BERT Similarity — сравнение пары предложений по смыслу
- Universal Sentence Encoder / LaBSE — когерентность между предложениями
"""

from sentence_transformers import util
import numpy as np
from functools import lru_cache
from typing import List, Dict
import os
os.environ["USE_TF"] = "0"               # ключевая строка
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


# --- Ленивая инициализация моделей/метрик (никаких глобальных объектов!) ---


@lru_cache(maxsize=1)
def _get_sbert():
    from sentence_transformers import SentenceTransformer
    # CPU инициализация стабильнее на macOS; можно сменить на "mps"/"cuda" при необходимости
    return SentenceTransformer("sentence-transformers/LaBSE", device="cpu")


@lru_cache(maxsize=1)
def _get_bertscore():
    import evaluate
    return evaluate.load("bertscore")

# ---------- BERTSCORE ----------


def calculate_bertscore(texts: List[str], lang: str = "ru") -> Dict:
    """
    BERTScore между парами (i, i+1)
    """
    bert_score = _get_bertscore()
    references = texts[:-1]
    candidates = texts[1:]
    result = bert_score.compute(
        predictions=candidates,
        references=references,
        lang=lang,
        # опционально: device="cpu", model_type="xlm-roberta-base"
    )

    return {
        "overall": {
            "precision": float(np.mean(result["precision"])),
            "recall": float(np.mean(result["recall"])),
            "f1": float(np.mean(result["f1"])),
        },
        "individual": list(zip(references, candidates, result["f1"])),
    }

# ---------- SEMANTIC SIMILARITY ----------


def calculate_semantic_similarity(texts: List[str]) -> Dict:
    """
    Semantic Textual Similarity (STS) между всеми парами текстов
    """
    sbert_model = _get_sbert()
    embeddings = sbert_model.encode(
        texts, convert_to_tensor=True, batch_size=8, normalize_embeddings=True)
    similarity_matrix = util.pytorch_cos_sim(
        embeddings, embeddings).cpu().numpy()

    scores = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            scores.append(float(similarity_matrix[i][j]))

    return {
        "overall_similarity_mean": float(np.mean(scores)),
        "pairwise": scores,
    }

# ---------- SBERT-BASED COHERENCE ----------


def calculate_coherence_sbert(texts: List[str]) -> Dict:
    """
    Оценка когерентности через Sentence-BERT (LaBSE) — сравнение соседних предложений
    """
    import re
    sbert_model = _get_sbert()
    scores = []

    for text in texts:
        sentences = re.split(r"[.!?]", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) < 2:
            scores.append(None)
            continue

        sent_embeddings = sbert_model.encode(
            sentences, convert_to_tensor=True, batch_size=8, normalize_embeddings=True)
        sims = util.pytorch_cos_sim(
            sent_embeddings[:-1], sent_embeddings[1:]).diagonal().cpu().numpy()
        scores.append(float(np.mean(sims)))

    valid = [s for s in scores if s is not None]
    return {
        "individual": scores,
        "overall": float(np.mean(valid)) if valid else float("nan"),
    }
