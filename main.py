import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings  # Adjust import based on the model
import numpy as np

# Load environment variables from .env at the project root
project_root = Path(__file__).resolve().parent
load_dotenv(project_root / ".env")

class GroqAPI:
    """Handles API operations with Groq to generate chat responses."""
    def __init__(self, model_name: str):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = model_name

    def _response(self, message):
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=message,
            temperature=0,
            max_tokens=4096,
            stream=True,
            stop=None,
        )

    def response_stream(self, message):        
        for chunk in self._response(message):
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

class Message:
    """Manages chat messages within the Streamlit UI."""
    system_prompt = "You are a professional AI. Please generate responses in English to all user inputs."

    def __init__(self):
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "system", "content": self.system_prompt}]

    def add(self, role: str, content: str):
        st.session_state.messages.append({"role": role, "content": content})

    def display_chat_history(self):
        for message in st.session_state.messages:
            if message["role"] == "system":
                continue
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def display_stream(self, generator):
        with st.chat_message("assistant"):
            return st.write_stream(generator)

class ModelSelector:
    """Allows the user to select a model from a predefined list."""
    def __init__(self):
        self.models = ["llama-3.2-1b-preview", "llama-3.2-3b-preview", "llama-3.2-11b-vision-preview", "llama-3.2-90b-vision-preview"]

    def select(self):
        with st.sidebar:
            st.sidebar.title("Groq Chat with Llama3 + α")
            return st.selectbox("Select a model:", self.models)

# Load the vector store using FAISS
def load_vector_store(vectorstore_name):
    vector_store_path = Path("vectorstore")  # Path to your FAISS index
    if vectorstore_name == "Faiss":
        # Create an instance of the embedding model
        embeddings = OpenAIEmbeddings()  # Replace with your specific embeddings class
        # Load the FAISS vector store
        db = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 40})
        return retriever
    else:
        raise ValueError("Unsupported vector store name")

def get_llm_response(llm, question, retriever):
    documents = retriever.get_relevant_documents(question)
    response = llm.response_stream(documents)
    return response

def main():
    user_input = st.text_input("Enter message to AI models...")
    model = ModelSelector()
    selected_model = model.select()

    message = Message()

    vectorstore_name = "Faiss"  # Adjust if necessary
    retriever = load_vector_store(vectorstore_name)

    if user_input:
        llm = GroqAPI(selected_model)
        message.add("user", user_input)
        message.display_chat_history()
        
        response = get_llm_response(llm, user_input, retriever)
        message.add("assistant", response)
        message.display_chat_history()

if __name__ == "__main__":
    main()
