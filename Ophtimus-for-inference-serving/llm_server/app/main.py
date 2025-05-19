from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.model_loader import create_ophtimus_chain

app = FastAPI()

# CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포 시 특정 origin만 허용하세요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    task: str
    instruction: str
    n: int = 2

@app.post("/generate")
def generate_answers(req: ChatRequest):
    chain = create_ophtimus_chain(req.task)
    answers = []
    for _ in range(req.n):
        stream = chain.stream({"instruction": req.instruction})
        result = "".join(token for token in stream)
        answers.append(result)
    return {"answers": answers}
