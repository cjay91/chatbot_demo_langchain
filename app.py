import streamlit as st
from playwright_file import ask


if 'generated' not in st.session_state:
    st.session_state['generated'] = []
# past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if msg is not None:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

user_input = st.chat_input("Ask Something")

if user_input:
    response = ""
    st.chat_message("user").markdown(user_input)
    response = ask(user_input)
    st.chat_message("assistant").markdown(response)

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})