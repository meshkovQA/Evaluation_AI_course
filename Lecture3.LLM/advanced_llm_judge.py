from ai_client import openai_chat_v2


def advanced_llm_judge(question, answer):
    """
    Продвинутая оценка с декомпозицией на утверждения.
    """

    # Шаг 1: Разбиваем ответ на утверждения
    decompose_prompt = f"""
Разбей этот ответ на отдельные проверяемые утверждения:
{answer}

Верни список утверждений, каждое с новой строки.
"""

    claims_text = openai_chat_v2(decompose_prompt)
    claims = [claim.strip()
              for claim in claims_text.split('\n') if claim.strip()]

    # Шаг 2: Оцениваем каждое утверждение
    scores = []
    verdict_values = {"fully": 1.0, "mostly": 0.9,
                      "partial": 0.6, "minor": 0.3, "none": 0.0}

    for claim in claims:
        evaluate_prompt = f"""
Оцени это утверждение относительно вопроса "{question}":
Утверждение: {claim}

Выбери один вариант:
- fully: полностью отвечает на вопрос
- mostly: в основном отвечает
- partial: частично релевантно
- minor: слабая связь
- none: нерелевантно

Верни только одно слово: fully/mostly/partial/minor/none
"""

        verdict = openai_chat_v2(evaluate_prompt).strip().lower()
        score = verdict_values.get(verdict, 0.6)
        scores.append(score)

    # Шаг 3: Считаем итоговую оценку
    final_score = sum(scores) / len(scores) if scores else 0
    final_score_10 = round(final_score * 10, 1)

    return {
        "score": final_score_10,
        "claims": claims,
        "individual_scores": scores,
        "verdict_distribution": {v: scores.count(s) for v, s in verdict_values.items()}
    }


# Пример использования
question = "Как работает фотосинтез?"
answer = "Фотосинтез -- это процесс, при котором растения используют свет для создания глюкозы из CO2 и воды. Происходит в хлоропластах. Выделяется кислород."

result = advanced_llm_judge(question, answer)
print(f"Продвинутая оценка: {result['score']}/10")
print(f"Утверждений: {len(result['claims'])}")
print(f"Распределение: {result['verdict_distribution']}")
