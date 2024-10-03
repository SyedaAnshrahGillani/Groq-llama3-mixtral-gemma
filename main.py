import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceHub

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
    system_prompt = "You are a professional AI assistant. Please generate responses based on the retrieved context and user input."

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
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def load_or_create_vectorstore(self):
        if self.index_path.exists():
            return FAISS.load_local(str(self.index_path), self.embeddings, allow_dangerous_deserialization=True)
        else:
            # Create a simple document if the index doesn't exist
            documents = ["This is a placeholder document for the FAISS index."]
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.create_documents(documents)
            vectorstore = FAISS.from_documents(texts, self.embeddings)
            vectorstore.save_local(str(self.index_path))
            return vectorstore

    def get_response(self, query: str, model_name: str):
        llm = HuggingFaceHub(repo_id=model_name, model_kwargs={"temperature": 0.5, "max_length": 512})
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(),
            memory=self.memory
        )
        return qa({"question": query})

def main():
    st.title("RAG-enhanced Groq Chat")

    model_selector = ModelSelector()
    selected_model = model_selector.select()

    message_manager = Message()
    rag = RAG("vectorstore")  # This will create the vectorstore if it doesn't exist

    user_input = st.text_input("Enter your question:")

    if user_input:
        message_manager.add("user", user_input)
        message_manager.display_chat_history()

        with st.spinner("Generating response..."):
            rag_response = rag.get_response(user_input, selected_model)
            groq_api = GroqAPI(selected_model)
            
            context_message = f"Context: {rag_response['source_documents']}\n\nHuman: {user_input}\n\nAI:"
            full_response = message_manager.display_stream(groq_api.response_stream([{"role": "user", "content": context_message}]))

        message_manager.add("assistant", full_response)

if __name__ == "__main__":
    main()
