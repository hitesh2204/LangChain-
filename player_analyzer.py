from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel,Field
from typing import Annotated
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

# creating the pydantic class
class Player(BaseModel):

    name :Annotated[str,Field(...,description ="name of player")]
    team :Annotated[str,Field(...,description ="name of team which player belongs too")]
    role :Annotated[str,Field(...,description ="role of player")]
    experience :Annotated[str,Field(...,description ="experience of player")]

# creating the model

model = ChatOpenAI(model="gpt-4o-mini",temperature=0)

# creating the structure model.

structure_model = model.with_structured_output(Player)

# creating the prompt.

prompt = PromptTemplate(
    template =""" Give me information about the cricket player {player}.

    Identify:
    - name
    - team
    - role
    - experience
    """,
    input_variables=['player']
)
parser = JsonOutputParser()

prompt1 = PromptTemplate(
    template ="""find out the information name,team,role and experience from given text {text}\n {format_instructions}""",
    input_variables=['text'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

convert_to_text = RunnableLambda(lambda x : x.model_dump_json())

chain = prompt | structure_model | convert_to_text | prompt1 | model | parser

result = chain.invoke({'player':'virat kohli'})
print(result)
