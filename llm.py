"""Логика вызова LLM через официальный клиент openai."""
from typing import Any

from openai import OpenAI

from config import settings

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


SYSTEM_PROMPT = (
    "Ты — сеньор-разработчик, который выражает свою точку зрения "
    "на разговорном гопническом сленге. Отвечай по сути инженерных "
    "вопросов — точно и по делу, но задорно, без занудства. "
    "Мат не обязателен, но лёгкая разговорная дерзость приветствуется. "
    "Пояснения держи короткими, фокус на решении."
)


def build_messages(system: str, history: list[dict]) -> list[dict]:
    """Собирает payload: системное сообщение + история чата."""
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.extend(history)
    return messages


def chat(messages: list[dict], temperature: float, max_tokens: int) -> str:
    """Выполняет запрос к chat/completions и возвращает текст ответа.

    Исключения openai/сети пробрасываются наверх для обработки в main.py.
    """
    response = get_client().chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""
