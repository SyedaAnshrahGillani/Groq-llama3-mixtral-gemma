from langchain.document_loaders import UnstructuredPDFLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains.retrieval import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import Chroma

## Introduction
loader = DirectoryLoader(path="data", glob="*.pdf", loader_class=UnstructuredPDFLoader)
documents = loader.load()

## Introduction
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
text_chunks = text_splitter.split_documents(documents)

## Introduction
embeddings = HuggingFaceEmbeddings()
vector_db = Chroma.from_documents(text_chunks, embeddings, persist_directory="Vector_DB_directory")
print("Documents vectorized.")