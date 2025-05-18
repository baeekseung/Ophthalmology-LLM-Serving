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
load_dotenv()

from langchain_teddynote import logging
logging.langsmith("Ophtimus-Web")

if not os.path.exists("./cache/"):
    os.makedirs("./cache/")

st.title("Ophthalmology Chatbot: Ophtimus")

if "Chat_History" not in st.session_state:
    st.session_state['Chat_History'] = []

with st.sidebar:
    clear_button = st.button("remove chat history")

    selected_task = st.selectbox(
    "Select model",
    ("Ophtimus Diagnosis", "Ophtimus Q&A"), index=0)

def print_chat_history():
    for chat_history in st.session_state.Chat_History:
        st.chat_message(chat_history.role).write(chat_history.content)

def add_message(role, message):
    st.session_state.Chat_History.append(ChatMessage(role=role, content=message))

def creaste_Ophtimus_chain(selected_task):
    if selected_task == "Ophtimus Diagnosis":
        prompt = load_prompt("prompts/Ophtimus_diagnosis.yaml", encoding="utf-8")

        hf = HuggingFacePipeline.from_model_id(
            model_id="MinWook1125/Opthimus_MCQA_EQA_CR_5000", 
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 512},
        )

        Ophtimus_model = ChatHuggingFace(llm=hf)

    elif selected_task == "Ophthalmology Q&A":
        tokenizer  = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        chat = [
            {"role": "system", "content": "You are an expert ophthalmologist. Please provide accurate and medically sound answers to the user's ophthalmology-related question."},
            {"role": "user",   "content": "{instruct}"},
        ]

        prompt_ = tokenizer.apply_chat_template(
            chat,
            tokenize=False,          # 문자열로 받기
            add_generation_prompt=True
        )
        prompt = PromptTemplate.from_template(prompt_)

        hf = HuggingFacePipeline.from_model_id(
            model_id="BaekSeungJu/Ophtimus-Llama-8B", 
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 512},
        )

        Ophtimus_model = ChatHuggingFace(llm=hf)

    chain = prompt | Ophtimus_model | StrOutputParser()
    return chain

if clear_button:
    st.session_state.Chat_History = []
    st.rerun()

print_chat_history()

user_input = st.chat_input("input your question")

if user_input:
    # 사용자 입력 출력
    st.chat_message("user").write(user_input)

    chain = creaste_Ophtimus_chain(selected_task)
    response = chain.stream({"instruction": user_input})

    with st.chat_message("assistant"):
        container = st.empty()
        ai_response = ""
        for token in response:
            ai_response += token
            container.markdown(ai_response)

    # 대화기록 저장 
    add_message("user", user_input)
    add_message("assistant", ai_response)

