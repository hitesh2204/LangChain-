from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

#loading openai embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

#loading csv file.
loader = CSVLoader("RAG//vector_store//player_performance.csv")
documents = loader.load()

player_doc = [document.page_content.replace('\n','|') for document in documents]
print("player doc leangth-",len(player_doc))

# creating the vector store.
vector_store = FAISS.from_texts(player_doc,embedding_model)

retriever = vector_store.as_retriever(search_kwargs={"k":2})

#creating the retriever.
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

query = "Which wicketkeeper-batsman has the highest strike rate?"
results = qa_chain.invoke({
    "query": query
})

print(results['result'])


