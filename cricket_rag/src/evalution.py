from cricket_rag.src.rag import RagPipeline
from cricket_rag.src.vector_store import VectoreStore 
from cricket_rag.src.retriever import Retriever

# Create vector store 
vector_store = VectoreStore()

# Create retriever 
retriever_class = Retriever()

# Load FAISS 
vector_db = vector_store.load_vector_store()

# Create retriever 
retriever = retriever_class.initiate_retriever(vector_db)

# Create RAG 
rag = RagPipeline()

# Evaluation question 
query = "What is a powerplay?"

# Run RAG 
result = rag.rag_pipeline(retriever, query)

# Get answer 
answer = result["result"]

# Get retrieved context 
contexts = [ doc.page_content for doc in result["source_documents"] ]

print("\nQUESTION:") 
print(query) 

print("\nANSWER:") 
print(answer) 

print("\nRETRIEVED CONTEXT:") 
for context in contexts: print(context) 
print("-" * 50)