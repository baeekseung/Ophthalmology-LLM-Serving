import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import ChatMessage
from langchain_teddynote.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_teddynote import logging
from langchain_teddynote.messages import stream_response
import mysql.connector
from datetime import datetime
from langchain_openai import ChatOpenAI

logging.langsmith("Ophtimus-Web")

load_dotenv()
os.environ["TRANSFORMERS_CACHE"] = "./cache/"
os.environ["HF_HOME"] = "./cache/"
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

st.title("Ophthalmology Chatbot: Ophtimus")

if "Chat_History" not in st.session_state:
    st.session_state["Chat_History"] = []

with st.sidebar:
    clear_button = st.button("remove chat history")

    selected_task = st.selectbox(
        "Select task", ("Ophthalmology Diagnosis", "Ophthalmology Q&A"), index=0
    )


def print_chat_history():
    for chat_history in st.session_state.Chat_History:
        st.chat_message(chat_history.role).write(chat_history.content)


def add_message(role, message):
    st.session_state.Chat_History.append(ChatMessage(role=role, content=message))


def Ophtimus_chain(selected_task):
    if selected_task == "Ophthalmology Diagnosis":
        prompt = load_prompt("prompts/Ophtimus_diagnosis.yaml", encoding="utf-8")

        # model_name = "MinWook1125/Opthimus_MCQA_EQA_CR_5000"
        # tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        # model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
        # pipe = pipeline("text-generation", model=model_name, tokenizer=tokenizer, max_new_tokens=512)
        # Ophtimus_model = HuggingFacePipeline(pipeline=pipe)

        # hf_endpoint_url = (
        #     "https://vsjtsipqov7izenf.us-east-1.aws.endpoints.huggingface.cloud"
        # )
        # Ophtimus_model = HuggingFaceEndpoint(
        #     # 엔드포인트 URL을 설정합니다.
        #     endpoint_url=hf_endpoint_url,
        #     max_new_tokens=1024,
        #     temperature=0.01,
        # )
        Ophtimus_model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.01,
        )

    elif selected_task == "Ophthalmology Q&A":
        prompt = load_prompt("prompts/Ophtimus_QA.yaml", encoding="utf-8")

        # model_name = "MinWook1125/Opthimus_MCQA_EQA_CR_5000"
        # tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        # model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
        # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
        # Ophtimus_model = HuggingFacePipeline(pipeline=pipe)

        # hf_endpoint_url = (
        #     "https://d2kp9g04i4a162pw.us-east-1.aws.endpoints.huggingface.cloud"
        # )
        # Ophtimus_model = HuggingFaceEndpoint(
        #     # 엔드포인트 URL을 설정합니다.
        #     endpoint_url=hf_endpoint_url,
        #     max_new_tokens=1024,
        #     temperature=0.01,
        # )

        Ophtimus_model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.01,
        )

    chain = prompt | Ophtimus_model | StrOutputParser()
    return chain


# MySQL 연결 설정
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="your_username",
        password="your_password",
        database="ophthalmology_qa",
    )


# 데이터베이스 테이블 생성
def create_tables():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS qa_responses (
            id INT AUTO_INCREMENT PRIMARY KEY,
            question TEXT,
            response1 TEXT,
            response2 TEXT,
            selected_response TEXT,
            created_at DATETIME
        )
    """
    )
    conn.commit()
    cursor.close()
    conn.close()


# 응답 저장 함수
def save_responses(question, response1, response2, selected_response):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO qa_responses (question, response1, response2, selected_response, created_at)
        VALUES (%s, %s, %s, %s, %s)
    """,
        (question, response1, response2, selected_response, datetime.now()),
    )
    conn.commit()
    cursor.close()
    conn.close()


# 초기화 시 테이블 생성
create_tables()

if clear_button:
    st.session_state.Chat_History = []
    st.rerun()

print_chat_history()

user_input = st.chat_input("input your question")

if user_input:
    # 사용자 입력 출력
    st.chat_message("user").write(user_input)

    chain = Ophtimus_chain(selected_task)

    # 두 개의 응답 생성
    response1 = ""
    response2 = ""

    with st.chat_message("assistant"):
        st.write("응답 1:")
        container1 = st.empty()
        for token in chain.stream({"instruction": user_input}):
            response1 += token
            container1.markdown(response1)

    with st.chat_message("assistant"):
        st.write("응답 2:")
        container2 = st.empty()
        for token in chain.stream({"instruction": user_input}):
            response2 += token
            container2.markdown(response2)

    # 응답 선택 UI
    st.write("어떤 응답이 더 적절한가요?")
    selected_response = st.radio(
        "응답 선택", ["응답 1", "응답 2"], key="response_selection"
    )

    # 선택된 응답 저장
    if selected_response == "응답 1":
        final_response = response1
    else:
        final_response = response2

    # 데이터베이스에 저장
    save_responses(user_input, response1, response2, final_response)

    # 대화기록 저장
    add_message("user", user_input)
    add_message("assistant", final_response)
