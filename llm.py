"""Логика вызова LLM через официальный клиент openai."""
import json
from typing import Any

from openai import OpenAI

from config import settings

# Сигнальный токен-ограничитель: модель должна оборвать ответ ровно на нём.
STOP_TOKEN = "КОНЕЦ"

# Явное описание формата ответа + условие завершения (добавляется к системному
# сообщению в «ограниченном» варианте сравнения). Стоп-слово нужно, чтобы
# stop-последовательность реально срабатывала: модель обязана завершить ответ
# словом КОНЕЦ, а inference-код обрежет его и всё, что после него.
CONSTRAINED_PROMPT = (
    "Требования к формату ответа: верни ТОЛЬКО валидный JSON без пояснений и "
    "без обрамляющих ```json```. Строго следуй схеме:\n"
    "{\n"
    '  "summary": "краткий тезис в одну строку",\n'
    '  "points": ["пункт 1", "пункт 2", "пункт 3"],\n'
    '  "action": "рекомендуемое следующее действие"\n'
    "}\n"
    "Не добавляй ничего до или после объекта — только чистый JSON.\n"
    "Условие завершения: сразу после закрывающей фигурной скобки напиши с "
    "новой строки ровно слово КОНЕЦ и остановись."
)

# Singleton: клиент создаётся один раз на всё время работы процесса
_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Возвращает единственный экземпляр OpenAI-клиента."""
    global _client
    if _client is None:
        kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        _client = OpenAI(**kwargs)
    return _client


def build_messages(system: str, history: list[dict]) -> list[dict]:
    """Собирает payload: (опциональное) системное сообщение + история чата."""
    messages: list[dict] = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.extend(history)
    return messages


def chat(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    stop: list[str] | None = None,
) -> str:
    """Выполняет запрос к chat/completions и возвращает текст ответа.

    Исключения openai/сети пробрасываются наверх для обработки в main.py.
    """
    return create_response(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
    )["content"]


def create_response(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    stop: list[str] | None = None,
) -> dict:
    """Выполняет запрос и возвращает словарь с текстом и метаданными ответа.

    Возвращает:
        content         — текст ответа
        finish_reason   — причина завершения ('stop', 'length', 'stop_sequence'…)
        prompt_tokens   — токены в запросе (или None, если usage недоступен)
        completion_tokens — токены в ответе (или None)
    """
    response = get_client().chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
    )
    choice = response.choices[0]
    usage = response.usage
    return {
        "content": choice.message.content or "",
        "finish_reason": choice.finish_reason,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
    }


def clean_content(content: str, stop: str | None = None) -> str:
    """Корректно обрабатывает сырой ответ модели.

    - срезает хвостовой стоп-токен (если inference его не вырезал сам);
    - снимает обрамляющие ```json```, если модель их всё-таки добавила.
    """
    text = (content or "").strip()
    if stop:
        s = stop.strip()
        if s and text.endswith(s):
            text = text[: -len(s)].strip()
    if text.startswith("```"):
        if text.endswith("```"):
            text = text[3:-3].strip()
        else:
            text = text[3:].strip()
        if text.startswith("json"):
            text = text[4:].strip()
    return text


def is_valid_json(content: str, stop: str | None = None) -> bool:
    """Проверяет, что ответ (после очистки) — валидный JSON."""
    try:
        json.loads(clean_content(content, stop))
        return True
    except (ValueError, TypeError):
        return False
