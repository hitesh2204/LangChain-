from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import CharacterTextSplitter

loader = CSVLoader("RAG//text_splitter//player.csv")

documents = loader.load()

print(len(documents))

splitter = CharacterTextSplitter(
    chunk_size = 30,
    chunk_overlap =0,
    separator="\n"
)

chunks = splitter.split_documents(documents)

print(len(chunks))
print(chunks)