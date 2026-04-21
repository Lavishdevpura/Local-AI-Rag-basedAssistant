# api_server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional          # ← was missing
import uvicorn

from rag.retriever import HybridRetriever
from voice.voice_routes import router as voice_router
from rag.upload_endpoint import router as upload_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def root():
    return FileResponse("index.html")

app.include_router(upload_router)
app.include_router(voice_router)

print("Loading retriever...")
retriever = HybridRetriever()
print("Ready.")

class ChatRequest(BaseModel):
    message: str
    confirmed: Optional[bool] = False   # ← added

class ChatResponse(BaseModel):
    response: str                        # ← was malformed before

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):             # ← keep req, not request
    result = retriever.query_agent(
        req.message,
        confirmed=req.confirmed          # ← pass confirmed, use req not request
    )
    return ChatResponse(response=result)

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)