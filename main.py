from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import AskRequest, AskResponse
from rag import ask
import traceback

app = FastAPI(title="Legal Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask_endpoint(payload: AskRequest):
    try:
        answer, sources = ask(payload.question)   # ✅ rag.ask returns Tuple[str, List[dict]]

        return {
            "answer": answer,
            "articles": None,
            "sources": sources
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
