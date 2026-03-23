FROM python:3.11-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Создаем пользователя для приложения
RUN useradd --create-home --shell /bin/bash app

# Устанавливаем рабочую директорию
WORKDIR /home/app

# Копируем файлы requirements
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY app/ app/
COPY .env.example .env

# Создаем директории для хранения данных с правильными правами
RUN mkdir -p storage/documents storage/vector_db/chroma && \
    chown -R app:app /home/app

# Переключаемся на пользователя приложения
USER app

# Открываем порт
EXPOSE 8002

# Переменные окружения
ENV PYTHONPATH=/home/app
ENV PYTHONUNBUFFERED=1

# Команда запуска
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]