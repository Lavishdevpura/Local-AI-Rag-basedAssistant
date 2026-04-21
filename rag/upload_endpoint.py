"""
upload_endpoint.py  –  Drop-in FastAPI router for document upload + auto-ingestion.

Mount this in your existing api_server.py:

    from upload_endpoint import router as upload_router
    app.include_router(upload_router)
"""

import os
import shutil
import asyncio
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# ── adjust this to wherever your ingest pipeline lives ──────────────────────
from config.settings import DOCUMENTS_DIR
from rag.ingestion import run_ingestion          # the function you already have
# ────────────────────────────────────────────────────────────────────────────

SUPPORTED = {".pdf", ".txt", ".md", ".docx"}

router = APIRouter()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    1. Validates file type.
    2. Saves the file to DOCUMENTS_DIR.
    3. Runs ingestion in a background thread so the HTTP response
       returns quickly while embedding/indexing happens asynchronously.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(SUPPORTED)}"
        )

    dest = Path(DOCUMENTS_DIR) / file.filename
    try:
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")

    # Run ingestion in a thread so we don't block the event loop
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, run_ingestion)

    return JSONResponse({
        "status": "ok",
        "message": f"'{file.filename}' uploaded. Ingestion started in background.",
        "filename": file.filename
    })


@router.get("/upload/status")
async def upload_status():
    """
    Light endpoint the frontend can poll to check whether the
    vector store is being updated (extend with real status flags as needed).
    """
    return {"status": "ready"}