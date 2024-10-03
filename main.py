import os  # To interact with the operating system and environment variables.
import streamlit as st  # To create and run interactive web applications directly through Python scripts.
from pathlib import Path  # To provide object-oriented filesystem paths, enhancing compatibility across different operating systems.
from dotenv import load_dotenv  # To load environment variables from a .env file into the system's environment for secure and easy access.
from groq import Groq  # To interact with Groq's API for executing machine learning models and handling data operations.
from langchain import FAISS  # Import FAISS for vector store handling
from langchain.chains import create_stuff_documents_chain, create_retrieval_chain  # Import relevant chain functions
from langchain_core.prompts import ChatPromptTemplate  # Import for prompt templates
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings  # For embeddings
import pickle  # To load the FAISS vector store

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
    system_prompt = "You are a professional AI. Please generate responses in English to all user inputs."

    # Initialize chat history if it doesn't exist in session state
    def __init__(self):
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "system", "content": self.system_prompt}]

    # Add a new message to the session state
    def add(self, role: str, content: str):
        st.session_state.messages.append({"role": role, "content": content})

    # Display all past messages in the UI, skipping system messages
    def display_chat_history(self):
        for message in st.session_state.messages:
            if message["role"] == "system":
                continue
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Stream API responses to the Streamlit chat message UI
    def display_stream(self, generator):
        with st.chat_message("assistant"):
            return st.write_stream(generator)

class ModelSelector:
    """Allows the user to select a model from a predefined list."""
    def __init__(self):
        # List of available models to choose from
        self.models = ["llama-3.2-1b-preview", "llama-3.2-3b-preview", "llama-3.2-11b-vision-preview", "llama-3.2-90b-vision-preview"]

    # Display model selection in a sidebar with a title
    def select(self):
        with st.sidebar:
            st.sidebar.title("Groq Chat with Llama3 + α")
            return st.selectbox("Select a model:", self.models)

# Load existing FAISS vector store from the specified directory
def load_vector_store():
    vector_store_path = Path("vectorstore/index.pkl")  # Adjust the path as necessary
    with open(vector_store_path, "rb") as f:
        vector_store = pickle.load(f)
    return vector_store

# Get response from the LLM using RAG logic
def get_llm_response(llm, prompt, question, vector_store):
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(vector_store.as_retriever(), document_chain)
    response = retrieval_chain.invoke({'input': question})
    return response

# Entry point for the Streamlit app
def main():
    user_input = st.text_input("Enter message to AI models...")
    model = ModelSelector()
    selected_model = model.select()

    message = Message()
    
    # Load vector store once
    vector_store = load_vector_store()

    # Prepare prompt for RAG
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based on the provided context only. If the question is not within the context, do not try to answer
        and respond that the asked question is out of context or something similar.
        Please provide the most accurate response based on the question.
        <context>
        {context}
        Questions: {input}
        """
    )

    # If there's user input, process it through the selected model
    if user_input:
        llm = GroqAPI(selected_model)
        message.add("user", user_input)
        message.display_chat_history()
        
        # Get response using RAG logic
        response = get_llm_response(llm, prompt, user_input, vector_store)
        
        message.add("assistant", response['answer'])
        message.display_chat_history()

if __name__ == "__main__":
    main()
