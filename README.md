# Чат с LLM (FastAPI + OpenAI)

Простой веб-интерфейс для общения с LLM через официальный клиент `openai`.

## Стек
- Backend: FastAPI + Uvicorn
- Frontend: одиночный HTML-файл с чистым JS (`fetch`)
- HTTP-клиент LLM: библиотека `openai` (chat/completions)

## Быстрый старт

### 1. Создайте окружение и установите зависимости
```
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

### 2. Настройте ключ API
Скопируйте шаблон в `.env` и впишите реальный ключ:
```
copy .env.example .env
```
Откройте `.env` и задайте как минимум `OPENAI_API_KEY=ваш-реальный-ключ`.
Опционально можно указать модель и базовый URL (например, для кастомного
шлюза/прокси):
```
OPENAI_MODEL=deepseek-v4-flash
OPENAI_BASE_URL=https://gpustack.data.lmru.tech/v1
```

> Ключ нигде не дублируется — он читается только из `.env` на сервере и не
> отправляется на клиент. Файл `.env` исключён из git.

### 3. Запустите сервер
```
uvicorn main:app --reload
```

### 4. Откройте в браузере
```
http://127.0.0.1:8000
```

## Структура проекта
- `main.py` — FastAPI-приложение и эндпоинты
- `config.py` — настройки (pydantic-settings из `.env`)
- `llm.py` — singleton-клиент OpenAI и логика вызова
- `static/index.html` — фронтенд
- `requirements.txt`, `.env.example`, `.gitignore`

## Эндпоинты
- `GET /` — HTML-страница
- `POST /api/chat` — принимает `{messages}` (обязательный массив сообщений)
  и опциональные `{temperature, max_tokens, system}` со значениями по умолчанию,
  возвращает `{"reply": "..."}`

## Важно
- История чата не хранится на сервере: на каждый запрос клиент присылает актуальный
  массив `messages`.
- Системный промпт захардкожен: модель отвечает как «сеньор-разработчик-гопник».
- Параметры `temperature` (0–2) и `max_tokens` (>0) валидируются на бэкенде;
  если не переданы, используются значения по умолчанию (0.7 и 1024).
