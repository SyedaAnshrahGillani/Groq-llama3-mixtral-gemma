import os
import PyPDF2
import streamlit as st
from io import StringIO
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import SVMRetriever
from groq.cloud.core import Completion
from PIL import Image

# Load and set the logo for the page
img = Image.open("img/dl_small.png")
st.set_page_config(page_title="Document QA", page_icon=img)

# Function to load documents (PDF or TXT)
@st.cache_data
def load_docs(files):
    st.info("`Reading document ...`")
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text += text
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text += text
        else:
            st.warning('Please upload only .txt or .pdf files.', icon="⚠️")
    return all_text

# Function to split text into chunks
@st.cache_resource
def split_texts(text, chunk_size, overlap):
    st.info("`Splitting document ...`")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    splits = text_splitter.split_text(text)
    if not splits:
        st.error("Failed to split document.")
        st.stop()
    return splits

# Function to create a retriever using FAISS or SVM
@st.cache_resource
def create_retriever(embeddings, splits, retriever_type):
    if retriever_type == "SIMILARITY SEARCH":
        try:
            vectorstore = FAISS.from_texts(splits, embeddings)
        except (IndexError, ValueError) as e:
            st.error(f"Error creating vector store: {e}")
            return
        retriever = vectorstore.as_retriever(k=5)
    elif retriever_type == "SUPPORT VECTOR MACHINES":
        retriever = SVMRetriever.from_texts(splits, embeddings)
    return retriever

# Function to query GroqAPI for final answer generation in Korean
def generate_answer_with_groq(user_query, retrieved_context):
    # Combine the retrieved information into a system prompt
    sys_prompt = f"""
    Instructions:
    - Be helpful and answer questions concisely.
    - Use the context provided to generate the answer.
    - Reply in Korean.
    Context: {retrieved_context}
    """
    # Send the user query and system prompt to Groq API
    with Completion() as completion:
        response, _, _ = completion.send_prompt("llama2-70b-4096", user_prompt=user_query, system_prompt=sys_prompt)
    return response

# Streamlit UI starts here
st.title("Document QA System")
st.write("Upload your documents and ask questions!")

uploaded_files = st.file_uploader("Upload your document (PDF or TXT)", accept_multiple_files=True)

if uploaded_files:
    loaded_text = load_docs(uploaded_files)
    st.write("Documents uploaded and processed.")

    # Split the document into chunks
    splits = split_texts(loaded_text, chunk_size=1000, overlap=0)
    num_chunks = len(splits)
    st.write(f"Number of text chunks: {num_chunks}")

    # Choose embedding model
    embedding_option = st.selectbox("Choose embedding model", ["OpenAI Embeddings", "HuggingFace Embeddings (slower)"])
    retriever_type = st.selectbox("Choose retriever type", ["SIMILARITY SEARCH", "SUPPORT VECTOR MACHINES"])

    # Embed using the selected embeddings
    if embedding_option == "OpenAI Embeddings":
        embeddings = OpenAIEmbeddings()
    elif embedding_option == "HuggingFace Embeddings (slower)":
        embeddings = HuggingFaceEmbeddings()

    retriever = create_retriever(embeddings, splits, retriever_type)

    if retriever:
        user_query = st.text_input("Ask your question here:")

        if user_query:
            # Perform retrieval
            st.info("`Retrieving relevant document chunks ...`")
            results = retriever.get_relevant_documents(user_query)
            context = " ".join([res.page_content for res in results])

            # Display retrieved chunks
            st.write("Retrieved context:")
            st.write(context)

            # Generate final answer using Groq
            st.info("`Generating answer ...`")
            final_answer = generate_answer_with_groq(user_query, context)
            st.write("Final Answer (in Korean):")
            st.write(final_answer)
