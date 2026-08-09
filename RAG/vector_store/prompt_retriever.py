from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
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

query = "Which wicketkeeper-batsman has the highest strike rate?"

context = retriever.invoke(query)

text = "\n".join(
    document.page_content
    for document in context
)

prompt = PromptTemplate(
    template ="""answer the questions by using context given below
        - context: {context}
        - query :{query}
    """,
    input_variables=['query','context']
)

chain = prompt | llm 
result = chain.invoke({'query':query,'context':text})
print(result.content)




