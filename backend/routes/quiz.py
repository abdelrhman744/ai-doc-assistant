from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

from services.rag_service import generate_quiz

router = APIRouter()


class QuizRequest(BaseModel):
    topic: str = ""
    num_questions: int = Field(default=5, ge=1, le=20)
    question_type: Literal["mcq", "true_false", "short_answer", "mixed"] = "mixed"
    language: Literal["auto", "ar", "en"] = "auto"


@router.post("/quiz/generate")
async def generate_quiz_route(request: QuizRequest):
    try:
        result = generate_quiz(
            topic=request.topic,
            num_questions=request.num_questions,
            question_type=request.question_type,
            lang=request.language,
        )

        return {
            "quiz": result["quiz"],
            "sources": result["sources"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))