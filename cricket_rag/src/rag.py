from cricket_rag.src.vector_store import VectoreStore
from cricket_rag.src.retriever import Retriever
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

# createing the llm
llm = ChatOpenAI(model="gpt-4o-mini",
                 temperature=0)


class RagPipeline:

    def __init__(self):
        pass

    def rag_pipeline(self, retriever):

        # Creating RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever
        )

        query = "What does the batting team try to achieve?"

        # Run RAG
        result = qa_chain.invoke({
            "query": query
        })

        return result


if __name__ == "__main__":

    # Create vector store object
    vector_store = VectoreStore()

    # Create retriever object
    retriever_class = Retriever()

    # Load existing FAISS vector store
    vector_db = vector_store.load_vector_store()

    # Create retriever
    retriever = retriever_class.initiate_retriever(vector_db)

    # Run RAG pipeline
    rag = RagPipeline()

    result = rag.rag_pipeline(retriever)

    print(result["result"])