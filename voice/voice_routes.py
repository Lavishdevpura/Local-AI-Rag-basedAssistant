# voice/voice_routes.py
#
# Add these routes to your existing FastAPI app (api_server.py):
#
#   from voice.voice_routes import router as voice_router
#   app.include_router(voice_router)

import asyncio
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from voice.speech_to_text import SpeechToText
from voice.text_to_speech import TextToSpeech

router = APIRouter()

# Singletons — initialised once at import time
_stt = SpeechToText()
_tts = TextToSpeech()


# /listen   POST
@router.post("/listen")
async def listen():
    """Record from mic and transcribe with Whisper. Blocking until silence or max_duration."""
    try:
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, _stt.transcribe)
        return {"text": text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"text": "", "error": str(e)})


# /listen/cancel   POST
@router.post("/listen/cancel")
async def listen_cancel():
    """Interrupt ongoing recording immediately — stops within 100ms."""
    _stt.cancel()
    return {"status": "cancelled"}


# /speak   POST  { "text": "..." }
class SpeakRequest(BaseModel):
    text: str

@router.post("/speak")
async def speak(req: SpeakRequest):
    """Speak text via pyttsx3 non-blocking."""
    if not req.text or not req.text.strip():
        return {"status": "empty"}
    _tts.speak_async(req.text)
    return {"status": "speaking"}


# /speak/stop   POST
@router.post("/speak/stop")
async def speak_stop():
    """Stop pyttsx3 speech immediately."""
    _tts.stop()
    return {"status": "stopped"}