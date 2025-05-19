import streamlit as st
import requests
from langchain_core.messages import ChatMessage

API_URL = "https://<YOUR_NGROK_URL>/generate"

st.title("Ophthalmology Chatbot: Ophtimus")

# 세션 초기화
for key in ["Chat_History", "generated_answers", "selected_answer", "current_question", "show_answers", "selected_idx"]:
    if key not in st.session_state:
        st.session_state[key] = [] if "answers" in key else None

with st.sidebar:
    st.markdown("## 대화기록 초기화")
    if st.button("remove chat history"):
        for key in st.session_state:
            st.session_state[key] = [] if isinstance(st.session_state[key], list) else None
        st.rerun()

    selected_task = st.selectbox("Select model", ("Ophtimus Diagnosis", "Ophtimus Q&A"))

def add_message(role, content):
    st.session_state.Chat_History.append(ChatMessage(role=role, content=content))

def print_chat_history():
    for chat in st.session_state.Chat_History:
        st.chat_message(chat.role).write(chat.content)

print_chat_history()

user_input = st.chat_input("input your question")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.current_question = user_input

    with st.spinner("답변 생성 중…"):
        res = requests.post(API_URL, json={
            "instruction": user_input,
            "task": selected_task,
            "n": 2
        })
        st.session_state.generated_answers = res.json()["answers"]
        st.session_state.show_answers = True

if st.session_state.show_answers and st.session_state.generated_answers:
    col1, col2 = st.columns(2)
    for idx, col in enumerate((col1, col2)):
        with col:
            st.subheader(f"답변 {idx + 1}")
            st.markdown(st.session_state.generated_answers[idx])

    selected_idx = st.radio(
        "더 도움이 된 답변을 선택해주세요:",
        options=[0, 1],
        format_func=lambda i: f"답변 {i + 1}",
        horizontal=True,
        key="answer_selection",
    )
    st.session_state.selected_idx = selected_idx

    if st.button("이 답변 선택", key="select_answer"):
        chosen = st.session_state.generated_answers[selected_idx]
        st.session_state.selected_answer = chosen
        add_message("user", st.session_state.current_question)
        add_message("assistant", f"**선택된 답변**\n\n{chosen}")
        st.session_state.show_answers = False
        st.session_state.generated_answers = []
        st.session_state.selected_idx = None
        st.success("선택이 저장되었습니다!")
        st.rerun()
