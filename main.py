"""FastAPI приложение «чат с LLM»."""
from pathlib import Path

import openai
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from config import settings
import llm

app = FastAPI(title="Чат с LLM")

STATIC_DIR = Path(__file__).parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"


class ChatRequest(BaseModel):
    messages: list[dict] = Field(default_factory=list)
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=1024)
    system: str = ""

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0 <= v <= 2:
            raise ValueError("temperature должна быть в диапазоне 0..2")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_tokens должен быть положительным")
        return v


class CompareRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    temperature: float = Field(default=0.7)
    # Настройки «ограниченного» ответа
    format_guide: str = Field(default=llm.CONSTRAINED_PROMPT)
    max_tokens: int = Field(default=300)
    stop: str = Field(default=llm.STOP_TOKEN)

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0 <= v <= 2:
            raise ValueError("temperature должна быть в диапазоне 0..2")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_tokens должен быть положительным")
        return v


class StabilityRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    repeats: int = Field(default=3)
    # Те же настройки «ограниченного» ответа, что и в /api/compare
    format_guide: str = Field(default=llm.CONSTRAINED_PROMPT)
    max_tokens: int = Field(default=300)
    stop: str = Field(default=llm.STOP_TOKEN)

    @field_validator("repeats")
    @classmethod
    def validate_repeats(cls, v: int) -> int:
        if not 1 <= v <= 20:
            raise ValueError("repeats должен быть в диапазоне 1..20")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_tokens должен быть положительным")
        return v


@app.get("/")
def index():
    return FileResponse(INDEX_FILE)


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY не задан. Создайте .env из .env.example",
        )

    # Собираем payload (системное сообщение + история)
    messages = llm.build_messages(req.system, req.messages)

    try:
        reply = llm.chat(
            messages=messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
    except openai.AuthenticationError as e:
        raise HTTPException(status_code=401, detail=f"Ошибка авторизации API: {e}")
    except openai.RateLimitError as e:
        raise HTTPException(status_code=429, detail=f"Слишком много запросов: {e}")
    except openai.APIError as e:
        raise HTTPException(status_code=502, detail=f"Ошибка API: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка: {e}")

    return {"reply": reply}


@app.post("/api/compare")
def api_compare(req: CompareRequest):
    """Отправляет один и тот же prompt на API дважды:
    - «без ограничений»: обычные параметры, без формата/длины/стопа;
    - «с ограничениями»: явный формат, лимит длины (max_tokens) и stop-последовательность.
    """
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY не задан. Создайте .env из .env.example",
        )

    user_msg = {"role": "user", "content": req.prompt}

    try:
        free = llm.create_response(
            messages=llm.build_messages("", [user_msg]),
            temperature=req.temperature,
            max_tokens=settings.default_max_tokens,
        )

        constrained_messages = [
            {"role": "system", "content": req.format_guide},
            user_msg,
        ]
        constrained = llm.create_response(
            messages=constrained_messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            stop=[req.stop] if req.stop.strip() else None,
        )
    except openai.AuthenticationError as e:
        raise HTTPException(status_code=401, detail=f"Ошибка авторизации API: {e}")
    except openai.RateLimitError as e:
        raise HTTPException(status_code=429, detail=f"Слишком много запросов: {e}")
    except openai.APIError as e:
        raise HTTPException(status_code=502, detail=f"Ошибка API: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка: {e}")

    return {
        "prompt": req.prompt,
        "unconstrained": {**free, "max_tokens": settings.default_max_tokens},
        "constrained": {
            **constrained,
            "max_tokens": req.max_tokens,
            "stop": req.stop,
            "format_guide": req.format_guide,
        },
    }


@app.post("/api/stability")
def api_stability(req: StabilityRequest):
    """Прогоняет один и тот же prompt с одними и теми же ограничениями N раз.

    Демонстрирует детерминированность формата: данные могут отличаться,
    но структура (JSON-схема) должна выдерживаться в каждом прогоне.
    """
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY не задан. Создайте .env из .env.example",
        )

    user_msg = {"role": "user", "content": req.prompt}
    messages = [{"role": "system", "content": req.format_guide}, user_msg]
    stop = [req.stop] if req.stop.strip() else None

    runs: list[dict] = []
    try:
        for i in range(1, req.repeats + 1):
            resp = llm.create_response(
                messages=messages,
                temperature=settings.default_temperature,
                max_tokens=req.max_tokens,
                stop=stop,
            )
            runs.append(
                {
                    "run": i,
                    "content": resp["content"],
                    "valid_json": llm.is_valid_json(resp["content"], req.stop),
                    "finish_reason": resp["finish_reason"],
                    "completion_tokens": resp["completion_tokens"],
                }
            )
    except openai.AuthenticationError as e:
        raise HTTPException(status_code=401, detail=f"Ошибка авторизации API: {e}")
    except openai.RateLimitError as e:
        raise HTTPException(status_code=429, detail=f"Слишком много запросов: {e}")
    except openai.APIError as e:
        raise HTTPException(status_code=502, detail=f"Ошибка API: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка: {e}")

    valid = sum(1 for r in runs if r["valid_json"])
    return {
        "prompt": req.prompt,
        "repeats": req.repeats,
        "runs": runs,
        "summary": {"total": len(runs), "valid_json": valid, "all_valid": valid == len(runs)},
    }
