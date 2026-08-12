from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Creating the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


def rag():
    return "Query sent to the rag_pipeline"


def analytical():
    return "Query sent to the analytical function"


# Creating Pydantic schema
class RouteQuery(BaseModel):
    route: str = Field(
        description="Choose either 'rag' or 'analytics'"
    )


# Creating prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a cricket query router.

        Choose 'rag' for general cricket knowledge.

        Choose 'analytics' for player, team or match statistics.
        """
    ),
    ("human", "{query}")
])


# Creating router
router = llm.with_structured_output(RouteQuery)

# Creating router chain
router_chain = prompt | router


# User query
query = "What was Virat Kohli's batting average in IPL 2024?"

# Run router
result = router_chain.invoke({
    "query": query
})

print(result)
print()
if result.route == 'rag':
    answer = rag()
if result.route == 'analytics':
    answer = analytical()

print(answer)