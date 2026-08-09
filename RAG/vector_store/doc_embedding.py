from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model ="text-embedding-3-small")


docs= [
     "Virat Kohli is a batsman",
    "Rohit Sharma is a batsman",
    "Jasprit Bumrah is a bowler"
]

embed_doc = embedding.embed_documents(docs)
print(embed_doc)
print(len(embed_doc[0]))