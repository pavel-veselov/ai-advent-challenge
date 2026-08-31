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
