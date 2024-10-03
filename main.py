import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from langchain import FAISS
from langchain.chains import ConversationalRetrievalChain
import pickle

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

def load_vector_store():
    vector_store_path = Path("vectorstore/faiss_index.pkl")  # Adjust the path as necessary
    with open(vector_store_path, "rb") as f:
        vector_store = pickle.load(f)
        
        # Ensure the loaded vector store is of the expected type
        if not isinstance(vector_store, FAISS):
            raise ValueError("Loaded vector store is not of type FAISS.")
    
    return vector_store

def get_llm_response(llm, question, vector_store):
    retrieval_chain = ConversationalRetrievalChain.from_llm(llm, vector_store.as_retriever())
    response = retrieval_chain.invoke({'input': question})
    return response

def main():
    user_input = st.text_input("Enter message to AI models...")
    model = ModelSelector()
    selected_model = model.select()

    message = Message()
    
    # Load vector store once
    vector_store = load_vector_store()

    if user_input:
        llm = GroqAPI(selected_model)
        message.add("user", user_input)
        message.display_chat_history()
        
        response = get_llm_response(llm, user_input, vector_store)
        
        message.add("assistant", response['answer'])
        message.display_chat_history()

if __name__ == "__main__":
    main()
