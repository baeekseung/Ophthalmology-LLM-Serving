import streamlit as st
import os
from transformers import AutoTokenizer
from langchain_core.messages import ChatMessage
from langchain_core.prompts import PromptTemplate
from langchain_teddynote.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from langchain_teddynote.messages import stream_response
from langchain_huggingface import ChatHuggingFace
from dotenv import load_dotenv

# 삭제
# ----------------------------
from langchain_openai import ChatOpenAI

# ----------------------------

load_dotenv()

from langchain_teddynote import logging as lc_logging

lc_logging.langsmith("Ophtimus-Web")

CACHE_DIR = "./cache/"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# 세션 상태 초기화
if "Chat_History" not in st.session_state:
    st.session_state["Chat_History"] = []

if "generated_answers" not in st.session_state:
    st.session_state["generated_answers"] = []

if "selected_answer" not in st.session_state:
    st.session_state["selected_answer"] = None

if "current_question" not in st.session_state:
    st.session_state["current_question"] = None

if "show_answers" not in st.session_state:
    st.session_state["show_answers"] = False

if "selected_idx" not in st.session_state:
    st.session_state["selected_idx"] = None

with st.sidebar:
    st.markdown("## 대화기록 초기화")
    clear_button = st.button("remove chat history")

    # 사용자가 모델(작업) 선택
    selected_task = st.selectbox(
        "Select model",
        ("Ophtimus Diagnosis", "Ophtimus Q&A"),
        index=0,
    )

    if clear_button:
        st.session_state.Chat_History = []
        st.session_state.generated_answers = []
        st.session_state.selected_answer = None
        st.session_state.current_question = None
        st.session_state.show_answers = False
        st.session_state.selected_idx = None
        st.rerun()


def print_chat_history():
    for chat in st.session_state.Chat_History:
        st.chat_message(chat.role).write(chat.content)


def add_message(role: str, content: str):
    st.session_state.Chat_History.append(ChatMessage(role=role, content=content))


def create_ophtimus_chain(task: str):
    if task == "Ophtimus Diagnosis":
        prompt = load_prompt("prompts/Ophtimus_diagnosis.yaml", encoding="utf-8")

        # hf = HuggingFacePipeline.from_model_id(
        #     model_id="MinWook1125/Opthimus_MCQA_EQA_CR_5000",
        #     task="text-generation",
        #     pipeline_kwargs={
        #         "max_new_tokens": 512,
        #         "do_sample": True,
        #         "temperature": 0.9,
        #         "top_p": 0.95,
        #     },
        # )
        # llm = ChatHuggingFace(llm=hf)

        # 삭제
        # ----------------------------
        llm = ChatOpenAI(
            temperature=0.9,
            model_name="gpt-4o",  # 모델명
        )
        # ----------------------------

    elif task == "Ophtimus Q&A":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        chat_template = [
            {
                "role": "system",
                "content": "You are an expert ophthalmologist. Please provide accurate and medically sound answers to the user's ophthalmology-related question.",
            },
            {"role": "user", "content": "{instruction}"},
        ]
        prompt_text = tokenizer.apply_chat_template(
            chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt = PromptTemplate.from_template(prompt_text)

        # hf = HuggingFacePipeline.from_model_id(
        #     model_id="BaekSeungJu/Ophtimus-Llama-8B",
        #     task="text-generation",
        #     pipeline_kwargs={
        #         "max_new_tokens": 512,
        #         "do_sample": True,
        #         "temperature": 0.9,
        #         "top_p": 0.95,
        #     },
        # )
        # llm = ChatHuggingFace(llm=hf)

        # 삭제
        # ----------------------------
        llm = ChatOpenAI(
            temperature=0.9,
            model_name="gpt-4o",  # 모델명
        )
        # ----------------------------

    chain = prompt | llm | StrOutputParser()
    return chain


def generate_multiple_answers(question: str, chain, n: int = 2):
    answers = []
    for _ in range(n):
        response_stream = chain.stream({"instruction": question})
        answer = "".join(token for token in response_stream)
        answers.append(answer)
    return answers


st.title("Ophthalmology Chatbot: Ophtimus")
print_chat_history()

user_input = st.chat_input("input your question")

if user_input:
    # 사용자 질문 표시
    st.chat_message("user").write(user_input)
    st.session_state.current_question = user_input

    # 체인 생성 & 두 개의 답변 받기
    chain = create_ophtimus_chain(selected_task)
    with st.spinner("답변 생성 중…"):
        answers = generate_multiple_answers(user_input, chain, n=2)
        st.session_state.generated_answers = answers
        st.session_state.show_answers = True

if st.session_state.show_answers and st.session_state.generated_answers:
    # 두 개 답변 나란히 표시
    col1, col2 = st.columns(2)
    for idx, col in enumerate((col1, col2)):
        with col:
            st.subheader(f"답변 {idx + 1}")
            st.markdown(st.session_state.generated_answers[idx])

    # 사용자 선택
    selected_idx = st.radio(
        "더 도움이 된 답변을 선택해주세요:",
        options=[0, 1],
        format_func=lambda i: f"답변 {i + 1}",
        horizontal=True,
        key="answer_selection",
    )
    st.session_state.selected_idx = selected_idx

    if st.button("이 답변 선택", key="select_answer"):
        chosen_answer = st.session_state.generated_answers[
            st.session_state.selected_idx
        ]
        st.session_state.selected_answer = chosen_answer
        add_message("user", st.session_state.current_question)
        add_message("assistant", f"**선택된 답변**\n\n{chosen_answer}")
        st.session_state.show_answers = False
        st.session_state.generated_answers = []
        st.session_state.selected_idx = None
        st.success("선택이 저장되었습니다!")
        st.rerun()
