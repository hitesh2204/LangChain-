from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


class RagPipeline:

    def rag_pipeline(self, retriever, query):

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        result = qa_chain.invoke({
            "query": query
        })

        return result