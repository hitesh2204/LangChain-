from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("RAG//text_splitter//cricket.txt")

documents = loader.load()
print(len(documents))

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 50,
    chunk_overlap = 0
)

chunks = splitter.split_documents(documents)
print(len(chunks))
print()
print(chunks)