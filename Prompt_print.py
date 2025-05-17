from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage
import os
from langchain_teddynote import logging
logging.langsmith("Ophtimus-Web")

# 사용할 모델의 저장소 ID를 설정합니다.
repo_id = "meta-llama/Llama-3.1-8B-Instruct"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,  # 모델 저장소 ID를 지정합니다.
    max_new_tokens=256,  # 생성할 최대 토큰 길이를 설정합니다.
    temperature=0.1,
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],  # 허깅페이스 토큰
)

# ❷ Chat 래퍼 생성
chat_llm = ChatHuggingFace(llm=llm)

# ❸ LangChain 메시지 객체 작성
msgs = [
    SystemMessage(content="당신은 정중한 한국어 조교입니다."),
    HumanMessage(content="LangChain에서 chat template을 어떻게 써?"),
]

# ❹ invoke() 호출 → 내부에서 apply_chat_template로 포맷 후 LLM 호출
response = chat_llm.invoke(msgs)
print(response.content)
