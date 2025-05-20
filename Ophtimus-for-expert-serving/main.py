import streamlit as st
import requests

st.title("Ophthalmology Chatbot: Ophtimus")

if "Chat_History" not in st.session_state:
    st.session_state['Chat_History'] = []

with st.sidebar:
    clear_button = st.button("remove chat history")

    selected_task = st.selectbox(
        "Select model",
        ("Ophtimus Diagnosis", "Ophtimus Q&A"), index=0)

def add_message(role, message):
    from langchain_core.messages import ChatMessage
    st.session_state.Chat_History.append(ChatMessage(role=role, content=message))

def print_chat_history():
    for chat_history in st.session_state.Chat_History:
        st.chat_message(chat_history.role).write(chat_history.content)

if clear_button:
    st.session_state.Chat_History = []
    st.rerun()

print_chat_history()

user_input = st.chat_input("Input your question")

if user_input:
    st.chat_message("user").write(user_input)

    # FastAPI 서버에 POST 요청
    response = requests.post(
        "http://<YOUR_PUBLIC_URL>/chat",  # ngrok 주소 또는 포트포워딩 주소
        json={"instruction": user_input, "task": selected_task}
    )

    # st.write("✅ 서버 응답 상태코드:", response.status_code)
    # st.write("📦 서버 응답 본문:", response.text)

    result = response.json()
    ai_response = result["response"]

    with st.chat_message("assistant"):
        st.markdown(ai_response)

    add_message("user", user_input)
    add_message("assistant", ai_response)
