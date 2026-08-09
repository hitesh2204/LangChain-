from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

loader = TextLoader("RAG//vector_store//cricket.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size= 100,
    chunk_overlap = 0
)
document = splitter.split_documents(documents)

vector_store = FAISS.from_documents(
    document,
    embedding_model
)

query = "What is the purpose of the fielding side?"

results = vector_store.similarity_search(query,k=2)

for result in results:
    print(result.page_content)



