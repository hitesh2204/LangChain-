from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from dotenv import load_dotenv

load_dotenv()

embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')

loader = CSVLoader("RAG//vector_store//player.csv")
documents = loader.load()
#print(documents)
print(len(documents))

text =[]
for document in documents:
    text.append(document.page_content.replace('\n','|'))
print(text)

csv_embedding = embedding_model.embed_documents(text)
print(csv_embedding)
print(len(csv_embedding))
