from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda,RunnableBranch
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def player_name(data):
        return "This is player query"

def team(data):
   return "this is team query"      

player_name = RunnableLambda(player_name)
team_name  = RunnableLambda(team)

model = ChatOpenAI(model="gpt-4o-mini",temperature=0)

player_prompt = PromptTemplate(
      template = "Give me the information about cricket player {questions}",
      input_variables=['questions']
)

team_prompt = PromptTemplate(
      template = "Give me the informarion about team {questions}",
      input_variables=['questions']
)

parser = StrOutputParser()

player = RunnableBranch(
     (lambda x:x['type']=='player',RunnableLambda(lambda x :{"questions":x["questions"]}) | player_prompt | model | parser),
     (lambda x:x['type']=='team',RunnableLambda(lambda x:{"questions":x["questions"]}) | team_prompt | model | parser),
     RunnableLambda(lambda x:"Other query")
)

result = player.invoke({
      'type':'player',
      "questions":"what is sachin tendulkar highest run score in odi"
    })

print(result)