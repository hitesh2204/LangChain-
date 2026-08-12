from cricket_rag.src.rag import RagPipeline
from cricket_rag.src.vector_store import VectoreStore
from cricket_rag.src.retriever import Retriever

from datasets import Dataset

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
    answer_relevancy
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


# --------------------------------------------------
# 1. RAGAS evaluator LLM
# --------------------------------------------------

evaluator_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )
)


# --------------------------------------------------
# 2. RAGAS evaluator embeddings
# --------------------------------------------------

evaluator_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
)


# --------------------------------------------------
# 3. Create RAG
# --------------------------------------------------

vector_store = VectoreStore()

retriever_class = Retriever()

vector_db = vector_store.load_vector_store()

retriever = retriever_class.initiate_retriever(vector_db)

rag = RagPipeline()


# --------------------------------------------------
# 4. Evaluation question
# --------------------------------------------------

query = "What is a powerplay?"


# --------------------------------------------------
# 5. Ground truth
# --------------------------------------------------

ground_truth = (
    "A powerplay is a period in limited-overs cricket "
    "with specific fielding restrictions."
)


# --------------------------------------------------
# 6. Run RAG
# --------------------------------------------------

result = rag.rag_pipeline(
    retriever,
    query
)


# --------------------------------------------------
# 7. Get answer and contexts
# --------------------------------------------------

answer = result["result"]

contexts = [
    doc.page_content
    for doc in result["source_documents"]
]


# --------------------------------------------------
# 8. Create RAGAS dataset
# --------------------------------------------------

data = {
    "user_input": [query],
    "retrieved_contexts": [contexts],
    "response": [answer],
    "reference": [ground_truth]
}

dataset = Dataset.from_dict(data)


# --------------------------------------------------
# 9. Configure RAGAS metrics
# --------------------------------------------------

for metric in [
    faithfulness,
    context_precision,
    context_recall,
    answer_relevancy
]:
    metric.llm = evaluator_llm
    metric.embeddings = evaluator_embeddings


# --------------------------------------------------
# 10. Evaluate
# --------------------------------------------------

evaluation_result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        context_precision,
        context_recall,
        answer_relevancy
    ]
)


# --------------------------------------------------
# 11. Print results
# --------------------------------------------------

print("\nRAGAS EVALUATION")
print(evaluation_result)

