from langchain_core.runnables import RunnableBranch, RunnableLambda


def analyze_virat(data):
    return "Analyze Virat"


def analyze_other(data):
    return "Analyze another player"


player = RunnableBranch(
    (
        lambda x: x["player"] == "virat kohli",
        RunnableLambda(analyze_virat)
    ),
    RunnableLambda(analyze_other)
)


result = player.invoke({
    "player": "Virat Kohli"
})

print(result)