from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from cricket_rag.src.ingestion import DataIngestion
from dotenv import load_dotenv

load_dotenv()

# creating embedding
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

class VectoreStore():

    def __init__(self):
        pass

    def create_vector_store(self,chunks):

        # creating the vector store.
        vector_store = FAISS.from_documents(chunks,embedding_model)

        vector_store.save_local(
            "cricket_rag/vector_db"
        )

        return vector_store

    def load_vector_store(self):

        vector_store = FAISS.load_local(
            "cricket_rag/vector_db",
            embedding_model,
            allow_dangerous_deserialization=True
        )

        return vector_store


if __name__=="__main__":
    data = "cricket_rag//data//cricket_basics.txt"
    ingestion = DataIngestion()
    chunks = ingestion.initiate_data_ingestion(data)
    vector = VectoreStore()
    doc_vector = vector.create_vector_store(chunks)
    print("Vector store created successfully!")





