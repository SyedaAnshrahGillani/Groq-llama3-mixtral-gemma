import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq  # Groq API for interacting with models
import faiss  # For similarity search with FAISS
import pickle  # To load the vector store
from langchain.prompts import PromptTemplate  # Template for handling prompt with context

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
    system_prompt = "You are a professional AI. Please generate responses in Korean to all user inputs."

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
        self.models = ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]

    def select(self):
        with st.sidebar:
            st.sidebar.title("Groq Chat with Llama3 + α")
            return st.selectbox("Select a model:", self.models)

class FAISSHandler:
    """Handles FAISS-based similarity search using prebuilt index and vectors."""
    def __init__(self, vector_store_path):
        self.vector_store_path = vectorstorepath
        self.index = self._load_faiss_index()
        self.documents = self._load_documents()

    def _load_faiss_index(self):
        index_path = os.path.join(self.vector_store_path, "faiss.index")
        return faiss.read_index(index_path)

    def _load_documents(self):
        with open(os.path.join(self.vector_store_path, "documents.pkl"), "rb") as f:
            return pickle.load(f)

    def search(self, query_vector, top_k=3):
        """Perform similarity search on the vector store."""
        distances, indices = self.index.search(query_vector, top_k)
        return [self.documents[i] for i in indices[0]]

def main():
    st.title("Korean RAG Chatbot")

    # Input user message
    user_input = st.text_input("Enter your question...")

    # Select model
    model_selector = ModelSelector()
    selected_model = model_selector.select()

    # Initialize message handling
    message = Message()

    # Load FAISS and vector store
    vectorstorepath = "./vectorstore"
    faiss_handler = FAISSHandler(vector_store_path)

    # If the user enters a query
    if user_input:
        # 1. Convert user query to vector (this assumes vectorization is already handled)
        # For demo purposes, assume it's done and we use some dummy vector here
        query_vector = faiss.vector_to_array(faiss_handler.index)

        # 2. Perform similarity search on the FAISS index
        relevant_docs = faiss_handler.search(query_vector)

        # 3. Prepare the prompt for the language model
        context = "\n".join(relevant_docs)
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="Context: {context}\nQuestion: {question}\nAnswer in Korean:"
        )
        prompt = prompt_template.render(context=context, question=user_input)

        # 4. Send the final prompt to GroqAPI and get response
        llm = GroqAPI(selected_model)
        message.add("user", user_input)
        message.display_chat_history()
        response = message.display_stream(llm.response_stream([{"role": "user", "content": prompt}]))
        message.add("assistant", response)

if __name__ == "__main__":
    main()
