from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ---------------- CLASSIFIER ----------------

prompt = PromptTemplate(
    template="""
    Determine what type of cricketing question it is.

    Return only one of these:
    - player
    - team
    - other

    Question: {question}
    """,
    input_variables=["question"]
)

parser = StrOutputParser()

classifier = prompt | model | parser


# ---------------- PLAYER / TEAM PROMPTS ----------------

player_prompt = PromptTemplate(
    template="Give me the information about cricket player {question}",
    input_variables=["question"]
)

team_prompt = PromptTemplate(
    template="Give me the information about cricket team {question}",
    input_variables=["question"]
)


# ---------------- PRESERVE QUESTION + CLASSIFICATION ----------------

def classify_question(data):

    question = data["question"]

    question_type = classifier.invoke({
        "question": question
    })

    return {
        "question": question,
        "type": question_type.strip().lower()
    }


router_input = RunnableLambda(classify_question)


# ---------------- ROUTER ----------------

player_chain = (
    RunnableLambda(lambda x: {"question": x["question"]})
    | player_prompt
    | model
    | parser
)

team_chain = (
    RunnableLambda(lambda x: {"question": x["question"]})
    | team_prompt
    | model
    | parser
)


router = RunnableBranch(
    (lambda x: x["type"] == "player", player_chain),
    (lambda x: x["type"] == "team", team_chain),
    RunnableLambda(lambda x: "Other query")
)


# ---------------- FINAL CHAIN ----------------

chain = router_input | router


result = chain.invoke({
    "question": "Ind vs Aus ODI record?"
})

print(result)