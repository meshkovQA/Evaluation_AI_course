#!/usr/bin/env python3
"""
Скрипт для принудительной повторной обработки документов
Запускать внутри контейнера или с доступом к API
"""

import sys
import os
import time

# Добавляем путь к модулям приложения
sys.path.insert(0, '/home/app')

def force_reprocess_documents():
    """Принудительно переобрабатывает все документы"""
    print("=" * 70)
    print("🔄 ПРИНУДИТЕЛЬНАЯ ПЕРЕОБРАБОТКА ДОКУМЕНТОВ")
    print("=" * 70)
    
    try:
        # Импортируем необходимые модули
        from app.services.document_service import DocumentService
        from app.models.document import DocumentStatus
        import asyncio
        
        # Создаем сервис
        service = DocumentService()
        
        print(f"\n📚 Найдено документов: {len(service.documents)}")
        
        if not service.documents:
            print("⚠️  Нет документов для обработки")
            return
        
        # Переобрабатываем каждый документ
        for doc_id, document in service.documents.items():
            print(f"\n{'='*70}")
            print(f"📄 Документ: {document.title}")
            print(f"   ID: {doc_id}")
            print(f"   Текущий статус: {document.status.value}")
            print(f"   Файл: {document.file_path}")
            print(f"   Чанков: {len(document.chunks)}")
            
            if document.status == DocumentStatus.READY and len(document.chunks) > 0:
                print("   ✅ Документ уже обработан, пропускаем")
                continue
            
            print(f"   🔄 Начало обработки...")
            
            # Сбрасываем статус
            document.status = DocumentStatus.PROCESSING
            document.chunks = []
            document.content = ""
            
            # Запускаем обработку синхронно
            try:
                asyncio.run(service._process_document(doc_id))
                
                # Проверяем результат
                time.sleep(1)  # Даем время на завершение
                
                if document.status == DocumentStatus.READY:
                    print(f"   ✅ Успешно обработан!")
                    print(f"   📊 Создано чанков: {len(document.chunks)}")
                elif document.status == DocumentStatus.ERROR:
                    print(f"   ❌ Ошибка: {document.error_message}")
                else:
                    print(f"   ⚠️  Статус: {document.status.value}")
                    
            except Exception as e:
                print(f"   ❌ Исключение при обработке: {e}")
                import traceback
                traceback.print_exc()
        
        # Итоговая статистика
        print(f"\n{'='*70}")
        print("📊 ИТОГОВАЯ СТАТИСТИКА")
        print(f"{'='*70}")
        
        total = len(service.documents)
        ready = sum(1 for d in service.documents.values() if d.status == DocumentStatus.READY)
        error = sum(1 for d in service.documents.values() if d.status == DocumentStatus.ERROR)
        processing = sum(1 for d in service.documents.values() if d.status == DocumentStatus.PROCESSING)
        
        print(f"Всего документов: {total}")
        print(f"✅ Готовых: {ready}")
        print(f"❌ С ошибками: {error}")
        print(f"🔄 В обработке: {processing}")
        
        # Показываем документы с ошибками
        if error > 0:
            print(f"\n❌ Документы с ошибками:")
            for doc in service.documents.values():
                if doc.status == DocumentStatus.ERROR:
                    print(f"   - {doc.title}: {doc.error_message}")
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


def check_environment():
    """Проверяет окружение перед обработкой"""
    print("\n🔍 ПРОВЕРКА ОКРУЖЕНИЯ")
    print("=" * 70)
    
    # Проверяем OPENAI_API_KEY
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print(f"✅ OPENAI_API_KEY установлен: {openai_key[:10]}...{openai_key[-4:]}")
    else:
        print(f"❌ OPENAI_API_KEY не установлен!")
        print("   Установите в .env файле или переменных окружения")
        return False
    
    # Проверяем директории
    storage_path = os.getenv('DOCUMENTS_STORAGE_PATH', '/home/app/storage/documents')
    vector_path = os.getenv('CHROMA_PERSIST_DIRECTORY', '/home/app/storage/vector_db/chroma')
    
    print(f"📂 Хранилище документов: {storage_path}")
    print(f"   Существует: {os.path.exists(storage_path)}")
    
    print(f"📂 Векторная БД: {vector_path}")
    print(f"   Существует: {os.path.exists(vector_path)}")
    
    return True


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "ПЕРЕОБРАБОТКА ДОКУМЕНТОВ" + " " * 28 + "║")
    print("╚" + "=" * 68 + "╝")
    
    if not check_environment():
        print("\n❌ Окружение не готово. Исправьте проблемы и попробуйте снова.")
        sys.exit(1)
    
    force_reprocess_documents()
    
    print("\n✅ Готово! Проверьте документы через API:")
    print("   GET http://localhost:8002/api/v1/documents/")
    print()