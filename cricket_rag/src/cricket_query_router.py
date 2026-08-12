from cricket_rag.src.rag_router import RagPipeline
from cricket_rag.src.vector_store import VectoreStore
from cricket_rag.src.retriever import Retriever

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# Router schema
# -------------------------

class RouteQuery(BaseModel):

    route: str = Field(
        description="Choose either 'rag' or 'analytics'"
    )

# -------------------------
# Router LLM
# -------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

router = llm.with_structured_output(RouteQuery)

# -------------------------
# Router prompt
# -------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a cricket query router.

        Choose 'rag' for general cricket knowledge.

        Choose 'analytics' for player, team,
        match or numerical statistics.
        """
    ),
    ("human", "{query}")
])

router_chain = prompt | router

# -------------------------
# Create RAG
# -------------------------

vector_store = VectoreStore()

retriever_class = Retriever()

vector_db = vector_store.load_vector_store()

retriever = retriever_class.initiate_retriever(vector_db)

rag = RagPipeline()

# -------------------------
# User query
# -------------------------

query = "What was Virat Kohli's batting average in IPL 2024?"

# -------------------------
# Route query
# -------------------------

route = router_chain.invoke({
    "query": query
})

print("ROUTE:", route.route)

# -------------------------
# Execute selected route
# -------------------------

if route.route == "rag":

    result = rag.rag_pipeline(
        retriever,
        query
    )

    print("\nANSWER:")
    print(result["result"])

elif route.route == "analytics":

    print("\nSend query to analytics function")