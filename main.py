import os
import json
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.retrievers import Chroma
from langchain.memory import ConversationBufferMemory

## Introduction
with open('config.json') as f:
    config_data = json.load(f)
    os.environ["GRO_API_KEY"] = config_data["GRO_API_KEY"]

## Introduction
st.set_page_config(page_title="Multi Document RAG Chatbot", page_icon="?")

## Introduction
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = load_vector_store()
if 'chain' not in st.session_state:
    st.session_state.chain = create_chain(st.session_state.vector_store)

## Introduction
user_input = st.chat_input("Ask a question:")
if user_input:
    st.session_state.chat_history.append(('role': 'user', 'content': user_input))

    # Get response from the chain
    response = st.session_state.chain(('question': user_input))
    st.session_state.chat_history.append(('role': 'assistant', 'content': response['answer']))

## Introduction
for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])