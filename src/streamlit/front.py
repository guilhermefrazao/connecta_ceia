import streamlit as st
from langchain_openai import ChatOpenAI
from mongoengine import connect
from dotenv import load_dotenv

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.abspath(os.path.join(current_dir, os.pardir)))

sys.path.append(parent_dir)

load_dotenv()

from src.assistants.interaction import Assistant


db = os.getenv('CONNECTA_DB_NAME')
host = os.getenv('CONNECTA_MONGO_URI')

connect(db=db, host=host)

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0,
                                 openai_api_key=os.getenv('OPENAI_API_KEY'))

assistant = Assistant(llm=llm)

st.header("Chatbot Conecta CEIA", divider="violet")

#
rag_tables = st.toggle("Habilitar RAG Tabelas")
rag_video = st.toggle("Habilitar RAG Video")
log = st.toggle("Habilitar histórico")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Escreva uma mensagem"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = assistant.rag_chain(
        message=prompt, history=st.session_state.messages,
        rag_tables=rag_tables,
        rag_video=rag_video,
        log=log
    )
    print("response_assistant",response)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})