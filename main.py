import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

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
            st.sidebar.title("Groq Chat with Llama3 + RAG")
            return st.selectbox("Select a model:", self.models)

class RAG:
    """Handles Retrieval-Augmented Generation using FAISS."""
    def __init__(self, index_path: str, embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.index_path = Path(index_path)
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        self.vectorstore = self.load_or_create_vectorstore()

    def load_or_create_vectorstore(self):
        if self.index_path.exists():
            try:
                return FAISS.load_local(str(self.index_path), self.embeddings)
            except Exception as e:
                st.error(f"Error loading existing index: {e}. Creating a new one.")
                return self.create_new_vectorstore()
        else:
            return self.create_new_vectorstore()

    def create_new_vectorstore(self):
        documents = [
            "This is a placeholder document for the FAISS index.",
            "Add more relevant documents here to populate your knowledge base."
        ]
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        vectorstore = FAISS.from_documents(texts, self.embeddings)
        vectorstore.save_local(str(self.index_path))
        return vectorstore

    def get_relevant_context(self, query: str, k: int = 2):
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            st.error(f"Error during similarity search: {e}")
            return []

def main():
    st.title("RAG-enhanced Groq Chat with Llama3")

    model_selector = ModelSelector()
    selected_model = model_selector.select()

    message_manager = Message()
    rag = RAG("vectorstore")

    user_input = st.text_input("Enter your question:")

    if user_input:
        message_manager.add("user", user_input)
        message_manager.display_chat_history()

        with st.spinner("Generating response..."):
            relevant_contexts = rag.get_relevant_context(user_input)
            context = "\n".join(relevant_contexts) if relevant_contexts else "No relevant context found."
            
            groq_api = GroqAPI(selected_model)
            
            context_message = f"Context: {context}\n\nHuman: {user_input}\n\nAI:"
            full_response = message_manager.display_stream(groq_api.response_stream([{"role": "user", "content": context_message}]))

        message_manager.add("assistant", full_response)

if __name__ == "__main__":
    main()
