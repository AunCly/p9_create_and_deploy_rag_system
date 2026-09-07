from ragas import Dataset, experiment
from ragas.metrics import DiscreteMetric
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision

questions = [
    'A quelle date peut-on voir Dany-Io le sculpteur de pierre ?',
    'Nous sommes le 3 septembre, quel est le prochain concert à venir à la Sirène ?',
    'A quelle heure commence le spectacle Premier regard sur L’homme qui danse ?',
    'Quels sont les évènements gratuits en Septembre 2027 ?',
    'Quand se passe le festival de la BD d\'Angoulême en 2026 ?',
]

ground_truths = [
    'Il vous sera possible de voir Dany-Io le sculpteur de pierre du 5 au 7 juin 2026.',
    'Le prochain concert qui aura lieu à la Sirène sera FFF + 1ÈRE PARTIE le 02 octobre 2026 à 20h00.',
    'Le spectacle "Premier regard sur L’homme qui danse" commencera à 18h30.',
    'Je ne peux pas répondre à cette question car je ne connais que les évènements de l\'année 2026.',
    'Je ne peux pas répondre à cette question car je ne connais que les évènements à venir à La Rochelle.',
]

def create_ragas_dataset(questions, ground_truths) -> Dataset:
    dataset = Dataset(name="hf_doc_qa_eval", backend="local/csv", root_dir=".")

    for question, ground_truth in zip(questions, ground_truths):
        dataset.append({"question": question, "expected_answer": ground_truth})

    dataset.save()
    return dataset

faithfullness_scorer = Faithfulness(llm=llm)
answer_relevancy_scorer = AnswerRelevancy(llm=llm, embeddings=embeddings)
context_recall_scorer = ContextRecall(llm=llm)
context_precision_scorer = ContextPrecision(llm=llm)



@experiment()
async def evaluate_rag(row, rag, llm) -> Dict[str, Any]:
    """
    Run RAG evaluation on a single row.

    Args:
        row: Dictionary containing question and expected_answer
        rag: Pre-initialized RAG instance
        llm: Pre-initialized LLM client for evaluation

    Returns:
        Dictionary with evaluation results
    """

    # Evaluate correctness asynchronously
    faithfullness_score = await faithfullness_scorer.ascore(
        user_input=row['user_input'],
        expected_answer=row["expected_answer"],
        response=model_response,
        llm=llm
    )



    # Return evaluation results
    result = {
        **row,
        "model_response": model_response,
        "correctness_score": score.value,
        "correctness_reason": score.reason,
        "mlflow_trace_id": rag_response.get("mlflow_trace_id", "N/A"),  # MLflow trace ID for debugging (explained later)
        "retrieved_documents": [
            doc.get("content", "")[:200] + "..." if len(doc.get("content", "")) > 200 else doc.get("content", "")
            for doc in rag_response.get("retrieved_documents", [])
        ]
    }

    return result