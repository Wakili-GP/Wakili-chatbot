from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import AskRequest, AskResponse, SourceItem
from rag import ask
import traceback

app = FastAPI(title="Legal Assistant API")

# If you have a frontend on another port/domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change to your frontend domain later
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
        return ask(payload.question)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
