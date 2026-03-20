"""Jarvis server — API + frontend."""
from __future__ import annotations
import os
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from agent import chat

app = FastAPI(title="Jarvis", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        reply = await chat(req.message)
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

static_dir = Path(__file__).parent / "static"

@app.get("/")
async def index():
    return FileResponse(static_dir / "index.html")

def main():
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
