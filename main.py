import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import ChatMessage
from langchain_teddynote.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

load_dotenv()
os.environ["TRANSFORMERS_CACHE"] = "./cache/"
os.environ["HF_HOME"] = "./cache/"
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

st.title("Ophthalmology Chatbot: Ophtimus")

if "Chat_History" not in st.session_state:
    st.session_state['Chat_History'] = []

with st.sidebar:
    clear_button = st.button("remove chat history")

    selected_task = st.selectbox(
    "Select task",
    ("Ophthalmology Diagnosis", "Ophthalmology Q&A"), index=0)

def print_chat_history():
    for chat_history in st.session_state.Chat_History:
        st.chat_message(chat_history.role).write(chat_history.content)

def add_message(role, message):
    st.session_state.Chat_History.append(ChatMessage(role=role, content=message))

def Ophtimus_chain():
    if selected_task == "Ophthalmology Diagnosis":
        prompt = load_prompt("prompts/Ophtimus_diagnosis.yaml", encoding="utf-8")

        model_name = "MinWook1125/Opthimus_MCQA_EQA_CR_5000"
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
        Ophtimus_model = HuggingFacePipeline(pipeline=pipe)

    elif selected_task == "Ophthalmology Q&A":
        prompt = load_prompt("prompts/Ophtimus_Q&A.yaml", encoding="utf-8")

        model_name = "MinWook1125/Opthimus_MCQA_EQA_CR_5000"
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
        Ophtimus_model = HuggingFacePipeline(pipeline=pipe)

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

    chain = Ophtimus_chain(selected_task)
    response = chain.stream({"question": user_input})

    with st.chat_message("assistant"):
        container = st.empty()
        ai_response = ""
        for token in response:
            ai_response += token
            container.markdown(ai_response)

    # 대화기록 저장 
    add_message("user", user_input)
    add_message("assistant", ai_response)

