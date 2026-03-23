# rag_connector.py
import requests
import time
from typing import List, Dict, Any


class RAGConnector:
    """Простой коннектор к RAG системе"""

    def __init__(
        self,
        endpoint_url: str,
        api_key: str = None,
        timeout: int = 30
    ):

        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.timeout = timeout

    def query(self, question: str) -> Dict[str, Any]:

        try:
            # Заголовки
            headers = {'Content-Type': 'application/json'}
            if self.api_key:
                headers['X-API-Key'] = self.api_key

            # Запрос
            response = requests.post(
                self.endpoint_url,
                json={"message": question},
                headers=headers,
                timeout=self.timeout
            )

            response.raise_for_status()
            return response.json()

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return {'error': str(e)}

    def batch_query(
        self,
        questions: List[str],
        expected_answers: List[str] = None
    ) -> List[Dict[str, Any]]:

        results = []

        for i, question in enumerate(questions):
            print(f"📝 {i+1}/{len(questions)}: {question[:50]}...")

            # Запрос к RAG
            rag_response = self.query(question)

            # Формируем результат
            result = {
                'question': question,
                'answer': rag_response.get('content', ''),
                'contexts': [s.get('content', '') for s in rag_response.get('sources', [])],
                'ground_truth': expected_answers[i] if expected_answers else ''
            }

            results.append(result)
            time.sleep(0.1)  # Пауза между запросами

        print(f"✅ Обработано {len(results)} вопросов")
        return results


# # ==================== ПРИМЕР ИСПОЛЬЗОВАНИЯ ====================

# if __name__ == "__main__":

#     # Инициализируем коннектор
#     connector = RAGConnector(
#         endpoint_url="http://5.11.83.110:8002/api/v1/chat/",
#         api_key="rag-api-key"
#     )

#     # Вопросы
#     questions = [
#         'Какие методы используется в product_service?',
#         'Как работает процесс авторизации?'
#     ]

#     # Ожидаемые ответы
#     expected = [
#         'В процессе тестирования участвуют представители бизнес-подразделений, разработчики и тестировщики.',
#         'Разработчик оказывает помощь в прохождении тестовых сценариев, консультирует по подготовке данных.'
#     ]

#     # Получаем ответы
#     results = connector.batch_query(questions, expected)

#     print("\n=== Результаты ===")
#     for res in results:
#         print(f"\nВопрос: {res['question']}")
#         print(f"Ответ RAG: {res['answer']}")
#         print(f"Ожидаемый ответ: {res['ground_truth']}")
#         print(f"Контексты: {res['contexts']}")
