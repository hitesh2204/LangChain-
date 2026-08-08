from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("RAG//text_splitter//cricket.txt")

documents = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 50
)

chunks = splitter.split_documents(documents)
print(len(documents))
print(len(chunks))
print()

for chunk in chunks:
    print(chunk.page_content)
    print(chunk.metadata)