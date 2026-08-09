from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

#calling openai embedding model

embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')

# loading text documents.
loader = TextLoader("RAG//vector_store//football.txt")
documents = loader.load()
print("number of documents-",len(documents))

#creating chunk of that documents.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0
)

chunks = text_splitter.split_documents(documents)
print("Number of chunks-",len(chunks))

#creating embedding of that chunks and store in vector store
vector_store = FAISS.from_documents(chunks,embedding_model)

query = "Which Argentine player is famous for dribbling and passing?"
#creating the retriever.
retriever = vector_store.as_retriever(search_type='mmr',search_kwargs={"k":2})
results = retriever.invoke(query)

for result in results:
    print(result.page_content)
  

