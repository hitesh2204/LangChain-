from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda,RunnableParallel
from pydantic import BaseModel,Field
from dotenv import load_dotenv
from typing import Annotated
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# creating the Player pydantic object
class PlayerComparison(BaseModel):

    name :Annotated[str,Field(...,description="name of player")]
    team :Annotated[str,Field(...,description="team names")]
    role :Annotated[str,Field(...,description="role of player")]
    matches :Annotated[str,Field(...,description="number of matches played")]
    runs :Annotated[int,Field(...,description="Player runs")]

model = ChatOpenAI(model="gpt-4o-mini",temperature=0)

structure_model = model.with_structured_output(PlayerComparison)

prompt = PromptTemplate(
    template = "Give the details information about this player {player}",
    input_variables=['player']
)

parser = JsonOutputParser()

prompt1 = PromptTemplate(
    template = """ do comparison between two player base on name,role,teams,runs,matches and tell me who is best player{player1},{player2}\n{format_instructions}""",
    input_variables=['player1','player2'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

player = RunnableParallel({
    "player1":RunnableLambda(lambda x:{"player":x['player1']}) | prompt | structure_model,
    "player2":RunnableLambda(lambda x:{"player":x['player2']}) | prompt | structure_model
})

player_text = RunnableLambda(
    lambda x: {
        "player1": x["player1"].model_dump_json(),
        "player2": x["player2"].model_dump_json()
    }
)

chain = player | player_text | prompt1 | model | parser

result = chain.invoke({
    'player1':'virat kohli',
    'player2': 'sachin tendulkar'
})

print(result)
