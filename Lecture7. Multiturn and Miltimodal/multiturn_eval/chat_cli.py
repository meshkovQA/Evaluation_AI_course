"""
Интерактивный CLI чат для тестирования многоходовых бесед
с возможностью сохранения и оценки диалогов (DeepEval / Opik)
"""

import os
from agent_connector import AgentConnector
from conversation_storage import ConversationStorage

# Импорт из нового модуля evaluators
from evaluators import DEFAULT_CHATBOT_ROLE
from evaluators.deepeval import run_evaluation as run_deepeval_evaluation
from evaluators.deepeval import ProxyLLM as DeepEvalProxyLLM
from evaluators.opik import run_evaluation as run_opik_evaluation


# ============= КОНФИГУРАЦИЯ =============

# Настройки агента
AGENT_CONFIG = {
    "endpoint_url": "http://5.11.83.110:8005/ask",
    "api_key": "80456142-5441-4469-b97f-1d72b7802a93",
    "user_id": "AleksM"
}

# ============= НАСТРОЙКИ МОДЕЛИ ДЛЯ ОЦЕНКИ =============
# Выбор режима: "openai" (напрямую OpenAI) или "proxy" (через прокси)
MODEL_MODE = "proxy"  # "openai" или "proxy"

# Название модели
MODEL_NAME = "gpt-4o-mini"

# Настройки для прокси (если MODEL_MODE = "proxy")
PROXY_CONFIG = {
    "api_key": "sk-proxy-maEZp5Yp0-h9nOHDZXtoZPql5VRW3CqTqakKOQgsQtQ",
    "base_url": "http://5.11.83.110:8000"
}

# Порог для метрик
EVALUATION_THRESHOLD = 0.5

# Фреймворк по умолчанию: "deepeval" или "opik"
DEFAULT_FRAMEWORK = "deepeval"

# Название проекта для Opik
OPIK_PROJECT_NAME = "multiturn-evaluation"

# ============= АВТОМАТИЧЕСКАЯ НАСТРОЙКА ПРОКСИ =============
# Устанавливаем переменные окружения при запуске если используется прокси
if MODEL_MODE == "proxy":
    os.environ["OPENAI_API_KEY"] = PROXY_CONFIG["api_key"]
    os.environ["OPENAI_API_BASE"] = PROXY_CONFIG["base_url"]
    os.environ["OPIK_PROJECT_NAME"] = OPIK_PROJECT_NAME


def create_evaluation_model(framework: str):
    """
    Создаёт модель для оценки в зависимости от настроек
    """
    if MODEL_MODE == "openai":
        # Используем напрямую OpenAI (просто название модели)
        return MODEL_NAME

    elif MODEL_MODE == "proxy":
        if framework == "deepeval":
            # Создаём ProxyLLM для DeepEval
            return DeepEvalProxyLLM(
                model=MODEL_NAME,
                api_key=PROXY_CONFIG["api_key"],
                base_url=PROXY_CONFIG["base_url"]
            )
        elif framework == "opik":
            # Для Opik просто возвращаем название модели
            # Прокси настраивается через переменные окружения в run_evaluation()
            return MODEL_NAME

    return MODEL_NAME


def print_help():
    """Выводит справку по командам"""
    print("\n" + "=" * 60)
    print("📋 КОМАНДЫ:")
    print("=" * 60)
    print("  /new         - сохранить текущий диалог и начать новый")
    print("  /eval        - оценить диалоги (текущий фреймворк)")
    print("  /eval deepeval - оценить с DeepEval")
    print("  /eval opik   - оценить с Opik")
    print("  /framework   - показать/сменить фреймворк оценки")
    print("  /list        - показать список сохранённых диалогов")
    print("  /clear       - удалить все сохранённые диалоги")
    print("  /delete <id> - удалить диалог по ID")
    print("  /url <url>   - добавить URL для следующего запроса")
    print("  /urls        - показать текущие URLs")
    print("  /clearurls   - очистить URLs")
    print("  /help        - показать эту справку")
    print("  /exit        - выход из программы")
    print("=" * 60)


def print_banner():
    """Выводит баннер при запуске"""
    print("\n" + "=" * 60)
    print("🤖 MULTITURN CHAT EVALUATOR")
    print("   Интерактивный чат с оценкой многоходовых бесед")
    print("   Поддержка: DeepEval & Opik")
    print("=" * 60)
    print("\nВведите /help для списка команд")
    print("-" * 60)


def run_eval_with_framework(conversations, framework: str, threshold: float):
    """Запускает оценку с выбранным фреймворком"""

    # Создаём модель для оценки
    model = create_evaluation_model(framework)

    print(
        f"   Модель: {MODEL_NAME} ({'proxy' if MODEL_MODE == 'proxy' else 'OpenAI'})")

    if framework == "deepeval":
        run_deepeval_evaluation(
            conversations=conversations,
            model=model,
            threshold=threshold,
            chatbot_role=DEFAULT_CHATBOT_ROLE
        )
    elif framework == "opik":
        # Определяем нужен ли прокси для Opik
        use_proxy = MODEL_MODE == "proxy"
        run_opik_evaluation(
            conversations=conversations,
            model=model,
            verbose=True,
            project_name=OPIK_PROJECT_NAME,
            use_proxy=use_proxy,
            proxy_api_key=PROXY_CONFIG["api_key"] if use_proxy else None,
            proxy_base_url=PROXY_CONFIG["base_url"] if use_proxy else None
        )
    else:
        print(f"❌ Неизвестный фреймворк: {framework}")


def main():
    """Главная функция - интерактивный CLI чат"""

    print_banner()

    # Инициализация компонентов
    agent = AgentConnector(**AGENT_CONFIG)
    storage = ConversationStorage()

    # Состояние
    current_conversation = []
    current_urls = []
    current_framework = DEFAULT_FRAMEWORK

    print(f"\n🔗 Агент: {AGENT_CONFIG['endpoint_url']}")
    print(f"📁 Сохранённых диалогов: {storage.count()}")
    print(f"🆔 Сессия: {agent.session_id}")
    print(f"📊 Фреймворк оценки: {current_framework}")

    while True:
        try:
            # Ввод от пользователя
            user_input = input("\n\n👤 Вы: ").strip()

            if not user_input:
                continue

            # ============= КОМАНДЫ =============

            if user_input == "/help":
                print_help()
                continue

            elif user_input == "/new":
                # Сохранить текущий диалог и начать новый
                if current_conversation:
                    conv_id = storage.save_conversation(current_conversation)
                    print(f"\n✅ Диалог сохранён: {conv_id}")
                    print(f"   Всего диалогов: {storage.count()}")
                else:
                    print("\n⚠️  Текущий диалог пуст")

                current_conversation = []
                current_urls = []
                agent.new_session()
                print(f"🔄 Новый диалог начат (сессия: {agent.session_id})")
                continue

            elif user_input.startswith("/eval"):
                # Определяем фреймворк
                parts = user_input.split()
                if len(parts) > 1:
                    framework = parts[1].lower()
                else:
                    framework = current_framework

                # Сохранить текущий если есть
                if current_conversation:
                    save = input(
                        "Сохранить текущий диалог перед оценкой? (y/n): ").strip().lower()
                    if save == 'y':
                        conv_id = storage.save_conversation(
                            current_conversation)
                        print(f"✅ Диалог сохранён: {conv_id}")

                conversations = storage.load_all()
                if not conversations:
                    print("\n⚠️  Нет сохранённых диалогов для оценки")
                    continue

                print(
                    f"\n📊 Запуск оценки {len(conversations)} диалогов ({framework})...")

                run_eval_with_framework(
                    conversations=conversations,
                    framework=framework,
                    threshold=EVALUATION_THRESHOLD
                )
                continue

            elif user_input.startswith("/framework"):
                parts = user_input.split()
                if len(parts) > 1:
                    new_framework = parts[1].lower()
                    if new_framework in ["deepeval", "opik"]:
                        current_framework = new_framework
                        print(f"✅ Фреймворк изменён на: {current_framework}")
                    else:
                        print(f"❌ Неизвестный фреймворк: {new_framework}")
                        print("   Доступные: deepeval, opik")
                else:
                    print(f"\n📊 Текущий фреймворк: {current_framework}")
                    print("   Для смены: /framework deepeval или /framework opik")
                continue

            elif user_input == "/list":
                storage.list_conversations()
                continue

            elif user_input == "/clear":
                confirm = input(
                    "Удалить ВСЕ сохранённые диалоги? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    storage.clear()
                    print("✅ Все диалоги удалены")
                else:
                    print("❌ Отменено")
                continue

            elif user_input.startswith("/delete "):
                conv_id = user_input[8:].strip()
                if storage.delete(conv_id):
                    print(f"✅ Диалог {conv_id} удалён")
                else:
                    print(f"❌ Диалог {conv_id} не найден")
                continue

            elif user_input.startswith("/url "):
                url = user_input[5:].strip()
                if url:
                    current_urls.append(url)
                    print(f"✅ URL добавлен: {url}")
                    print(
                        f"   Текущие URLs ({len(current_urls)}): {current_urls}")
                continue

            elif user_input == "/urls":
                if current_urls:
                    print(f"\n🔗 Текущие URLs ({len(current_urls)}):")
                    for i, url in enumerate(current_urls, 1):
                        print(f"   {i}. {url}")
                else:
                    print("\n⚠️  URLs не заданы")
                continue

            elif user_input == "/clearurls":
                current_urls = []
                print("✅ URLs очищены")
                continue

            elif user_input == "/exit":
                # Предложить сохранить текущий диалог
                if current_conversation:
                    save = input(
                        "Сохранить текущий диалог перед выходом? (y/n): ").strip().lower()
                    if save == 'y':
                        conv_id = storage.save_conversation(
                            current_conversation)
                        print(f"✅ Диалог сохранён: {conv_id}")

                print("\n👋 До свидания!")
                break

            elif user_input.startswith("/"):
                print(f"❌ Неизвестная команда: {user_input}")
                print("   Введите /help для списка команд")
                continue

            # ============= ОБЫЧНОЕ СООБЩЕНИЕ =============

            # Показываем индикатор загрузки
            print("\n⏳ Отправка запроса...", end="", flush=True)

            # Отправляем запрос агенту
            urls_to_send = current_urls if current_urls else None
            response = agent.query(user_input, urls=urls_to_send)

            # Очищаем URLs после использования
            if current_urls:
                current_urls = []

            # Проверяем на ошибку
            if response.get('error'):
                print(f"\r❌ Ошибка: {response['error']}")
                continue

            # Извлекаем данные
            answer = response.get('output', '')
            tools = response.get('tools_used', [])

            # Выводим ответ
            print(f"\r🤖 Ассистент: {answer}")

            if tools:
                tools_str = ', '.join(tools)
                print(f"\n   [🔧 Tools: {tools_str}]")

            # Добавляем в текущий диалог
            current_conversation.append({
                "role": "user",
                "content": user_input
            })
            current_conversation.append({
                "role": "assistant",
                "content": answer,
                "tools_called": tools
            })

            # Показываем статус
            turns_count = len(current_conversation) // 2
            print(
                f"\n   [💬 Ход {turns_count} | Несохранённых сообщений: {len(current_conversation)}]")

        except KeyboardInterrupt:
            print("\n\n⚠️  Прервано пользователем (Ctrl+C)")
            if current_conversation:
                save = input(
                    "Сохранить текущий диалог? (y/n): ").strip().lower()
                if save == 'y':
                    conv_id = storage.save_conversation(current_conversation)
                    print(f"✅ Диалог сохранён: {conv_id}")
            print("👋 До свидания!")
            break

        except Exception as e:
            print(f"\n❌ Неожиданная ошибка: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
