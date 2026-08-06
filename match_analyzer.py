from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel,Field
from typing import Annotated
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import json

load_dotenv()

# creating the pydantic class
class MatchSummary(BaseModel):

    team1:Annotated[str,Field(...,description="Name of team1")]
    team2:Annotated[str,Field(...,description="name of team2")]
    team1_score:Annotated[int,Field(...,description="team1 score")]
    team2_score:Annotated[int,Field(...,description="team2 score")]
    winning_team:Annotated[str,Field(...,description="winning team name")]

# creating the model
model = ChatOpenAI(model="gpt-4o-mini",temperature=0)

structure_model = model.with_structured_output(MatchSummary)

def calculate_margin(match):

    if match.winning_team == match.team1:
        margin = match.team1_score - match.team2_score
        margin_type = "runs"

    else:
        margin = match.team2_score - match.team1_score
        margin_type = "runs"

    return {
        "winner": match.winning_team,
        "margin": margin,
        "margin_type": margin_type
    }

prompt = PromptTemplate(
    template = """ Give me the details summary about this match {summary}
    Identify:
        - team1
        - team2
        - team1 score 
        - team2 score
        - winning team
    """,
    input_variables=['summary']
)

win_margin = RunnableLambda(calculate_margin)

parser = JsonOutputParser()

prompt1 = PromptTemplate(
    template = " find the winner ,margin and margin type from given text {text}\n {format_instructions}",
    input_variables =['text'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

win_margin_text = RunnableLambda(lambda x : json.dumps(x))

chain = prompt | structure_model | win_margin | win_margin_text | prompt1 | model | parser

match_result = chain.invoke({'summary':'India scored 185/4 against Australia Australia scored 180/8'})

print(match_result)
