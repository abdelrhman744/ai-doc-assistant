from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Literal, Optional

from services.rag_service import ask_question
from services.audio_service import transcribe_audio

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    language: Literal["auto", "ar", "en"] = "auto"


@router.post("/chat")
async def chat(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        result = ask_question(request.query, lang=request.language)
        return {
            "answer":   result["answer"],
            "sources":  result["sources"],
            "stt_text": "",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/voice")
async def chat_voice(
    audio: UploadFile = File(...),
    language: str     = Form("auto"),
):
    """Transcribe uploaded audio then answer the question."""
    try:
        audio_bytes = await audio.read()
        lang_hint   = language if language in {"ar", "en"} else None
        stt_text    = transcribe_audio(audio_bytes, language=lang_hint)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Audio transcription failed: {e}")

    if not stt_text:
        raise HTTPException(
            status_code=422,
            detail="Whisper could not recognise any speech. Check microphone / audio quality.",
        )

    try:
        result = ask_question(stt_text, lang=language)
        return {
            "answer":   result["answer"],
            "sources":  result["sources"],
            "stt_text": stt_text,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
