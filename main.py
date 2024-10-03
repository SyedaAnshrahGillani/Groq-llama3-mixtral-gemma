from sentence_transformers import SentenceTransformer
import numpy as np

class FAISSHandler:
    """Handles FAISS-based similarity search using prebuilt index and vectors."""
    def __init__(self, vector_store_path):
        self.vector_store_path = vector_store_path
        self.index = self._load_faiss_index()
        self.documents = self._load_documents()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Example model

    def _load_faiss_index(self):
        index_path = os.path.join(self.vector_store_path, "index.faiss")
        return faiss.read_index(index_path)

    def _load_documents(self):
        with open(os.path.join(self.vector_store_path, "index.pkl"), "rb") as f:
            return pickle.load(f)

    def vectorize_query(self, query):
        """Convert a user query to a vector using the embedding model."""
        return self.embedding_model.encode([query])

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
    vector_store_path = "./vectorstore"  # Path to the vector store directory
    faiss_handler = FAISSHandler(vector_store_path)

    # If the user enters a query
    if user_input:
        # 1. Convert user query to vector using the embedding model
        query_vector = faiss_handler.vectorize_query(user_input)

        # 2. Perform similarity search on the FAISS index
        relevant_docs = faiss_handler.search(np.array(query_vector))

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
