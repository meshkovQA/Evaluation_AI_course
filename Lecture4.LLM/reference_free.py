# reference_free.py

"""
Reference-free метрики для оценки LLM
=====================================
Модуль использует несколько библиотек:
- perplexity - используем оценку через трансформеры с русской моделью
- textstat (для читаемости)
- language_tool_python (для грамматики)
- fast_bleu (для self-BLEU)
- diversity (для distinct-n)
- coherence для русского нужно использовать embeddings DeepPavlov/rubert-base-cased
"""

import numpy as np
from typing import List, Dict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from diversity import ngram_diversity_score
import textstat
import language_tool_python
import textdescriptives as td
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity


def calculate_perplexity(text: str, model_id: str = "sberbank-ai/rugpt3small_based_on_gpt2") -> float:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

    return torch.exp(loss).item()


def calculate_readability(text: str, lang: str = "ru") -> Dict[str, float]:
    textstat.set_lang(lang)
    return {
        "flesch": textstat.flesch_reading_ease(text)
    }


def calculate_grammar_errors(text: str, lang: str = "ru") -> int:
    tool = language_tool_python.LanguageTool(lang)
    return len(tool.check(text))


def calculate_coherence_with_transformers(text: str, transformer_model: str = "DeepPavlov/rubert-base-cased") -> Dict[str, float]:

    tokenizer = AutoTokenizer.from_pretrained(transformer_model)
    model = AutoModel.from_pretrained(transformer_model)
    model.eval()

    import re
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    def get_embedding(sent):
        inputs = tokenizer(sent, return_tensors="pt",
                           truncation=True, max_length=512)
        with torch.no_grad():
            return model(**inputs).last_hidden_state[:, 0, :].squeeze().numpy()

    embeddings = [get_embedding(s) for s in sentences]

    if len(embeddings) < 2:
        return {"first_order": None, "second_order": None}

    first_order = np.mean([
        cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
        for i in range(len(embeddings) - 1)
    ])
    second_order = np.mean([
        cosine_similarity([embeddings[i]], [embeddings[i+2]])[0][0]
        for i in range(len(embeddings) - 2)
    ]) if len(embeddings) > 2 else None

    return {
        "first_order": float(first_order),
        "second_order": float(second_order) if second_order else None
    }


def calculate_self_bleu(index: int, texts: List[str], n: int = 2) -> float:
    smoothing = SmoothingFunction().method1
    candidate = texts[index].split()
    references = [t.split() for i, t in enumerate(texts) if i != index]
    weights = tuple([1.0 / n] * n)
    return sentence_bleu(references, candidate, weights=weights, smoothing_function=smoothing)


def calculate_distinct(texts: List[str]) -> Dict[str, float]:
    distinct_1 = ngram_diversity_score(texts, 1)
    distinct_2 = ngram_diversity_score(texts, 2)
    return {
        "distinct_1": distinct_1,
        "distinct_2": distinct_2
    }
