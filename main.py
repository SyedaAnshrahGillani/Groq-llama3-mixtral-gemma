import os  # To interact with the operating system and environment variables.
import streamlit as st  # To create and run interactive web applications directly through Python scripts.
from pathlib import Path  # To provide object-oriented filesystem paths, enhancing compatibility across different operating systems.
from dotenv import load_dotenv  # To load environment variables from a .env file into the system's environment for secure and easy access.
from groq import Groq  # To interact with Groq's API for executing machine learning models and handling data operations.
import faiss  # For handling FAISS index for similarity search.
import pickle  # For loading your pre-trained model or data.
import numpy as np  # For numerical operations (e.g., handling vectors).
from langchain.embeddings import OpenAIEmbeddings  # Assuming you're using OpenAI embeddings (or replace with your own).
from langchain.vectorstores import FAISS  # For loading FAISS vector store.
from logger import log_runtime  # Import your logging decorator.

# Load environment variables from .env at the project root
project_root = Path(__file__).resolve().parent
load_dotenv(project_root / ".env")

class GroqAPI:
    """Handles API operations with Groq to generate chat responses."""
    def __init__(self, model_name: str):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = model_name

    # Internal method to fetch responses from the Groq API
    def _response(self, message):
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=message,
            temperature=0,
            max_tokens=4096,
            stream=True,
            stop=None,
        )

    # Generator to stream responses from the API
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
        self.models = ["llama-3.2-1b-preview", "llama-3.2-3b-preview", "llama-3.2-11b-vision-preview"]

    def select(self):
        with st.sidebar:
            st.sidebar.title("Groq Chat with Llama3 + α")
            return st.selectbox("Select a model:", self.models)

class RAG:
    """Handles Retrieval-Augmented Generation operations."""
    def __init__(self, vectorstore_path: str):
        # Load the FAISS vector store
        self.__embedding_function = OpenAIEmbeddings()  # Use your embedding function here
        self.db = FAISS.load_local(vectorstore_path, self.__embedding_function, allow_dangerous_deserialization=True)
        self.__base_retriever = self.db.as_retriever(search_kwargs={"k": 40})

    @log_runtime()
    async def retrieve_and_rerank(self, search_phrase: str):
        """Retrieve and rerank documents based on the search phrase."""
        documents = await self.retrieve_documents(search_phrase)
        if len(documents) == 0:  # to avoid empty API call
            return []
        
        docs = [doc.page_content for doc in documents if isinstance(doc, Document)]
        api_result = await self.make_rerank_api_call(search_phrase, docs)

        reranked_index = [res.index for res in api_result.results]
        from base_logger import logger
        logger.info(f"Cohere Reranking: {str(reranked_index)}")
        
        reranked_docs = []
        complete_text_tracker = []
        for res in api_result.results:
            doc = documents[res.index]
            if doc.metadata['complete_text'] in complete_text_tracker:
                continue
            documentItem = VectorStoreDocumentItem(
                page_content=doc.metadata['complete_text'],
                filename=doc.metadata['filename'],
                heading=doc.metadata['heading'],
                relevance_score=res.relevance_score
            )
            reranked_docs.append(documentItem)
            complete_text_tracker.append(doc.metadata['complete_text'])
            
        return reranked_docs

    async def retrieve_documents(self, search_phrase: str):
        """Retrieve documents based on a search phrase."""
        return await self.__base_retriever.retrieve(search_phrase)

    async def make_rerank_api_call(self, search_phrase: str, docs: list):
        """Call the API for reranking."""
        # Implement the API call logic here
        pass  # Replace with your API call logic.

# Entry point for the Streamlit app
def main():
    user_input = st.text_input("Enter message to AI models...")
    model_selector = ModelSelector()
    selected_model = model_selector.select()

    message = Message()

    # Initialize the RAG system with the path to the vector store
    rag = RAG("vectorstore")  # Update the path to your FAISS vector store

    if user_input:
        llm = GroqAPI(selected_model)

        # Retrieve and rerank relevant context based on user input
        retrieved_contexts = rag.retrieve_and_rerank(user_input)
        context = "\n".join([doc.page_content for doc in retrieved_contexts])  # Assuming retrieved_contexts is a list of document items.

        # Add context to the message for prompt engineering
        enhanced_message = [{"role": "system", "content": "Generate a response in Korean based on the following context:\n" + context},
                            {"role": "user", "content": user_input}]
        
        message.add("user", user_input)
        message.display_chat_history()
        response = message.display_stream(llm.response_stream(enhanced_message))
        message.add("assistant", response)

if __name__ == "__main__":
    main()
