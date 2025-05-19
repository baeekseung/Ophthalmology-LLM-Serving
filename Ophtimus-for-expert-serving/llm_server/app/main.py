from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.model_loader import load_model

app = FastAPI()

# CORS 허용 설정 (Streamlit Cloud에서 요청 가능하게)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 보안을 위해 실제 배포 시엔 특정 origin만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    task: str
    instruction: str

@app.post("/chat")
def chat(request: ChatRequest):
    chain = load_model(request.task)
    result = chain.invoke({"instruction": request.instruction})
    return {"response": result}
