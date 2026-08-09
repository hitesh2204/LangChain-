from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model ="text-embedding-3-small")

text = "I am Hitesh Yerekar"

embedding_doc = embedding.embed_query(text)
print(embedding_doc)
print()
print(len(embedding_doc))

