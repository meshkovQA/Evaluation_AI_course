# benchmarks.py
import re
import random
from datasets import load_dataset
from ai_client import openai_chat_v2


random.seed(42)

# =======================
# Вспомогательные функции
# =======================


def form_options(options: list):
    """Формирует список опций A, B, C..."""
    option_str = 'Options are:\n'
    for i, opt in enumerate(options):
        letter = chr(65 + i)
        option_str += f'({letter}): {opt}\n'
    return option_str


def get_prediction(output: str):
    """Извлекает ответ из текста модели в формате 'The answer is (X)'"""
    pattern = r"answer is\s*\(?([A-J])\)?"
    match = re.search(pattern, output, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    else:
        print("❌ Extraction failed, random guess")
        return random.choice([chr(65 + i) for i in range(10)])


def clean_response(text: str) -> str:
    """Очищает ответ модели от лишних символов"""
    if not text:
        return ""
    text = text.strip()
    # убираем все кроме букв и цифр
    text = re.sub(r"[^A-Za-z0-9]", "", text)
    return text.upper()


# =======================
# 1. MMLU-PRO
# =======================


def run_mmlu_pro(category='computer science', num_samples=10):
    print(f"\n=== MMLU-Pro: {category} ===")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")

    # Загружаем 5-shot примеры из validation
    cot_prefix = ''
    for d in dataset['validation']:
        if d['category'] == category:
            cot_prefix += 'Q: ' + \
                d['question'] + '\n' + \
                form_options(d['options']) + '\n' + d['cot_content'] + '\n\n'

    # Тестовая выборка
    test_set = [ex for ex in dataset['test']
                if ex['category'] == category][:num_samples]

    correct = 0

    for i, example in enumerate(test_set):
        question_block = 'Q: ' + example['question'] + \
            '\n' + form_options(example['options']) + '\n'
        full_prompt = cot_prefix + question_block

        model_output = openai_chat_v2(full_prompt)
        predicted = get_prediction(model_output)
        correct_answer = example['answer'].strip().upper()

        result = "✅" if predicted == correct_answer else "❌"
        print(f"{result} Q{i+1}: predicted={predicted}, correct={correct_answer}")

        if predicted == correct_answer:
            correct += 1

    accuracy = correct / num_samples
    print(f"🎯 MMLU-Pro Accuracy: {correct}/{num_samples} = {accuracy:.2f}")


# =======================
# 2. TruthfulQA
# =======================


def run_truthfulqa(num_samples=10):
    print("\n=== TruthfulQA ===")
    dataset = load_dataset("truthful_qa", "multiple_choice",
                           split="validation[:{}]".format(num_samples))

    correct = 0

    for i, example in enumerate(dataset):
        question = example["question"]
        choices = example["mc1_targets"]["choices"]
        # строка правильного ответа
        correct_choice = example["mc1_targets"]["labels"][0]

        prompt = f"{question}\nChoices:\n"
        for j, choice in enumerate(choices):
            prompt += f"{chr(65 + j)}. {choice}\n"
        prompt += "Answer:"

        model_output = openai_chat_v2(prompt)
        matched = any(correct_choice.strip().lower() in model_output.lower()
                      for correct_choice in example["mc1_targets"]["choices"])

        result = "✅" if matched else "❌"
        print(f"{result} Q{i+1}: response={model_output[:60]}...")

        if matched:
            correct += 1

    accuracy = correct / num_samples
    print(f"🎯 TruthfulQA Accuracy: {correct}/{num_samples} = {accuracy:.2f}")


# =======================
# 3. HellaSwag
# =======================

def run_hellaswag(num_samples=10):
    print("\n=== HellaSwag ===")
    dataset = load_dataset(
        "hellaswag", split="validation[:{}]".format(num_samples))

    correct = 0

    for i, example in enumerate(dataset):
        context = example["ctx"]
        endings = example["endings"]
        label = int(example["label"])

        prompt = f"{context}\nWhich ending is the most plausible?\n"
        for j, ending in enumerate(endings):
            prompt += f"{j}: {ending}\n"
        prompt += "Answer (provide number):"

        response = openai_chat_v2(prompt)
        digits = ''.join(filter(str.isdigit, response))
        prediction = int(digits) if digits.isdigit() else -1

        result = "✅" if prediction == label else "❌"
        print(f"{result} Q{i+1}: predicted={prediction}, correct={label}")

        if prediction == label:
            correct += 1

    accuracy = correct / num_samples
    print(f"🎯 HellaSwag Accuracy: {correct}/{num_samples} = {accuracy:.2f}")

# =======================
# 4. BIG-Bench Hard
# =======================


def run_bigbench_hard(task_name="date_understanding", num_samples=10):
    print(f"\n=== BIG-Bench Hard ({task_name}) ===")

    # Загрузка задачи и промптов
    dataset = load_dataset("Joschka/big_bench_hard",
                           task_name, split=task_name)
    prompts_ds = load_dataset("Joschka/big_bench_hard", "few_shot_prompts")
    prompt_data = prompts_ds['few_shot_prompts'].filter(
        lambda x: x['dataset_name'] == task_name
    )[0]
    # или chain_of_thought_prompt
    fewshot_prompt = prompt_data['answer_only_prompt']

    dataset = dataset.select(range(min(num_samples, len(dataset))))
    correct = 0

    for i, ex in enumerate(dataset):
        question = ex['question']
        choices = ex['choices']['text']
        labels = ex['choices']['label']  # e.g., ['A','B','C','D']
        target = ex['target']

        # Формируем prompt
        prompt = fewshot_prompt + "\nQ: " + question + "\n"
        for lbl, txt in zip(labels, choices):
            prompt += f"{lbl}. {txt}\n"
        prompt += "\nA:"

        response = openai_chat_v2(prompt)
        predicted = None
        for lbl in labels:
            if lbl.lower() in response.lower():
                predicted = clean_response(lbl)
                break

        result = "✅" if predicted == target else "❌"
        print(f"{result} Q{i+1}: predicted={predicted}, correct={target}")
        if result == "✅":
            correct += 1

    accuracy = correct / len(dataset)
    print(
        f"🎯 BIG-Bench Hard Accuracy: {correct}/{len(dataset)} = {accuracy:.2f}")


# =======================
# 5. MathQA
# =======================


def run_mathqa(num_samples=10):
    print("\n=== MathQA (Calc-X version) ===")
    dataset = load_dataset("MU-NLPC/calc-math_qa",
                           split=f"train[:{num_samples}]")

    correct = 0

    for i, ex in enumerate(dataset):
        question = ex["question"]
        options = ex["options"]  # словарь {"A": "...", ..., "E": "..."}
        answer = ex["result"]    # строка: "A", "B", ...

        # Формируем prompt
        prompt = question + "\n"
        for lbl, txt in options.items():
            prompt += f"{lbl}. {txt}\n"
        prompt += "Answer (letter):"

        response = openai_chat_v2(prompt)
        pred = response.strip()[0].upper()

        result = "✅" if pred == answer.upper() else "❌"
        print(f"{result} Q{i+1}: predicted={pred}, correct={answer.upper()}")

        if pred == answer.upper():
            correct += 1

    accuracy = correct / num_samples
    print(f"🎯 MathQA Accuracy: {correct}/{num_samples} = {accuracy:.2f}")


# =======================
# 6. BoolQ
# =======================

def run_boolq(num_samples=10):
    print("\n=== BoolQ ===")
    dataset = load_dataset("boolq", split=f"validation[:{num_samples}]")

    correct = 0

    for i, example in enumerate(dataset):
        question = example["question"]
        passage = example["passage"]
        answer = "yes" if example["answer"] else "no"

        prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer (yes or no):"

        response = openai_chat_v2(prompt)
        predicted = "yes" if "yes" in response.lower() else "no"

        result = "✅" if predicted == answer else "❌"
        print(f"{result} Q{i+1}: predicted={predicted}, correct={answer}")

        if predicted == answer:
            correct += 1

    accuracy = correct / num_samples
    print(f"🎯 BoolQ Accuracy: {correct}/{num_samples} = {accuracy:.2f}")


# =======================
# 7. LAMBADA
# =======================

def run_lambada(num_samples=10):
    print("\n=== LAMBADA ===")
    dataset = load_dataset("lambada", "plain_text",
                           split=f"validation[:{num_samples}]")

    correct = 0

    for i, example in enumerate(dataset):
        full_text = example["text"]
        parts = full_text.rsplit(" ", 1)
        context = parts[0]
        expected = parts[1]

        prompt = f"Complete the sentence: {context.strip()}\nNext word:"
        response = openai_chat_v2(prompt)
        prediction = clean_response(response.strip().split()[0])

        result = "✅" if expected.lower() == prediction.lower() else "❌"
        print(f"{result} Q{i+1}: predicted='{prediction}', expected='{expected}'")

        if expected.lower() == prediction.lower():
            correct += 1

    accuracy = correct / num_samples
    print(f"🎯 LAMBADA Accuracy: {correct}/{num_samples} = {accuracy:.2f}")


# =======================
# 8. Winogrande
# =======================

def run_winogrande(num_samples=10):
    print("\n=== Winogrande (automated-research-group/winogrande) ===")
    dataset = load_dataset(
        "automated-research-group/winogrande", split=f"validation[:{num_samples}]"
    )

    correct = 0

    for i, ex in enumerate(dataset):
        prompt = ex["request"]
        answer = ex["response"]

        response = openai_chat_v2(prompt)
        pred = ''.join(filter(str.isdigit, response.strip()))

        result = "✅" if pred == answer else "❌"
        print(f"{result} Q{i+1}: predicted={pred}, correct={answer}")

        if pred == answer:
            correct += 1

    accuracy = correct / num_samples
    print(f"🎯 Winogrande Accuracy: {correct}/{num_samples} = {accuracy:.2f}")


# =======================
# 9. IFEval
# =======================

def run_ifeval(num_samples=10):
    print("\n=== IFEval (google/IFEval) ===")
    dataset = load_dataset("google/IFEval", split=f"train[:{num_samples}]")

    for i, example in enumerate(dataset):
        prompt = example["prompt"]

        full_prompt = f"{prompt.strip()}\nAnswer:"
        response = openai_chat_v2(full_prompt)

        print(f"\n🔹 Q{i+1}: {prompt[:80]}...")
        print(f"🔸 Model: {response.strip()[:200]}")

    print(f"\n🎯 IFEval: completed {num_samples} generations.")


# =======================
# 10. BBQ
# =======================

def run_bbq(category=None, num_samples=10):
    print(f"\n=== BBQ ({category}) ===")

    dataset = load_dataset("Elfsong/BBQ", split=f"{category}[:{num_samples}]")

    correct = 0

    for i, example in enumerate(dataset):
        context = example["context"]
        question = example["question"]
        options = [example["ans0"], example["ans1"], example["ans2"]]
        correct_index = example["answer_label"]

        prompt = f"{context}\n{question}\n"
        for idx, option in enumerate(options):
            prompt += f"{idx}: {option}\n"
        prompt += "Answer with 0, 1 or 2:"

        response = openai_chat_v2(prompt)
        pred = ''.join(filter(str.isdigit, response.strip()))

        result = "✅" if pred and int(pred) == correct_index else "❌"
        print(f"{result} Q{i+1}: predicted={pred}, correct={correct_index}")

        if pred and int(pred) == correct_index:
            correct += 1

    accuracy = correct / num_samples
    print(f"🎯 BBQ Accuracy: {correct}/{num_samples} = {accuracy:.2f}")

# =======================
# 🚀 Запуск всех бенчмарков
# =======================


if __name__ == "__main__":
    run_mmlu_pro(category="computer science", num_samples=10)
    run_truthfulqa(num_samples=10)
    run_hellaswag(num_samples=10)
    run_bigbench_hard(task_name="date_understanding", num_samples=10)
    run_mathqa(num_samples=10)
    run_boolq(num_samples=10)
    run_lambada(num_samples=10)
    run_winogrande(num_samples=10)
    run_ifeval(num_samples=10)
    run_bbq(category="age", num_samples=10)
