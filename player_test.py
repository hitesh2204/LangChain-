from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnablePassthrough
from pydantic import BaseModel,Field
from typing import Annotated
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def player(data):
    return {
             "message": f"Analyzing {data['player']}"
            }

player_details = RunnableLambda(player)
parser = StrOutputParser()

player_summary = RunnableParallel({
    'original':RunnablePassthrough(),
    'modified':player_details
})

result = player_summary.invoke({'player':'Virat Kohli'})
print(result)

