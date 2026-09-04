import pandas as pd
from datasets import Dataset
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from script.rag import EventsRag as Rag
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.experiment import experiment

questions = [
    'A quelle date peut-on voir Dany-Io le sculpteur de pierre ?',
    # 'Nous sommes le 3 septembre, quel est le prochain concert à venir à la Sirène ?',
    # 'A quelle heure commence le spectacle Premier regard sur L’homme qui danse ?',
    # 'Quels sont les évènements gratuits en Septembre 2027 ?',
    # 'Quand se passe le festival de la BD d\'Angoulême en 2026 ?',
]

ground_truths = [
    'Il vous sera possible de voir Dany-Io le sculpteur de pierre du 5 au 7 juin 2026.',
    # 'Le prochain concert qui aura lieu à la Sirène sera FFF + 1ÈRE PARTIE le 02 octobre 2026 à 20h00.',
    # 'Le spectacle "Premier regard sur L’homme qui danse" commencera à 18h30.',
    # 'Je ne peux pas répondre à cette question car je ne connais que les évènements de l\'année 2026.',
    # 'Je ne peux pas répondre à cette question car je ne connais que les évènements à venir à La Rochelle.',
]

# Initialisation du RAG
rag = Rag()

@experiment()
def evaluate_rag(row, rag, llm, embeddings):

    # Query the RAG system
    rag_response = rag.query(row["question"], top_k=5)
    model_response = rag_response.get("answer", "")

    # Evaluate Faithfulness
    faithfullness_scorer = Faithfulness(llm=llm)
    faithfullness_result = faithfullness_scorer.score(
        user_input=row['question'],
        response=row['answer'],
        retrieved_contexts=row["retrieved_contexts"],
    )

    # Evaluate Answer Relevancy
    answer_relevancy_scorer = AnswerRelevancy(llm=llm, embeddings=embeddings)
    answer_relevancy_result = answer_relevancy_scorer.score(
        user_input=row['question'],
        response=row['answer'],
        retrieved_contexts=row["retrieved_contexts"],
    )

    # Evaluate Context Precision
    context_precision_scorer = ContextPrecision(llm=llm, embeddings=embeddings)
    context_precision_result = context_precision_scorer.score(
        user_input=row['question'],
        response=row['answer'],
        retrieved_contexts=row["retrieved_contexts"],
    )

    # Evaluate Context Recall
    context_recall_scorer = ContextRecall(llm=llm, embeddings=embeddings)
    context_recall_result = context_recall_scorer.score(
        user_input=row['question'],
        response=row['answer'],
        retrieved_contexts=row["retrieved_contexts"],
    )

    # Return evaluation results
    result = {
        **row,
        "model_response": model_response,
        "faithfullness_score": faithfullness_result.value,
        "faithfullness_reason": faithfullness_result.reason,
        "answer_relevancy_score": answer_relevancy_result.value,
        "answer_relevancy_reason": answer_relevancy_result.reason,
        "context_precision_score": context_precision_result.value,
        "context_precision_reason": context_precision_result.reason,
        "context_recall_score": context_recall_result.value,
        "context_recall_reason": context_recall_result.reason,
        "retrieved_documents": [
            doc.get("content", "")[:200] + "..." if len(doc.get("content", "")) > 200 else doc.get("content", "")
            for doc in rag_response.get("retrieved_documents", [])
        ]
    }

    return result

# 2. Préparation des données (inchangée)
answers = []
placeholder_contexts = []

for question in questions:
    answer, documents = rag.answer(question)
    answers.append(answer.content)

    if documents and hasattr(documents[0], 'page_content'):
        text_contexts = [doc.page_content for doc in documents]
    else:
        text_contexts = documents

    placeholder_contexts.append(text_contexts)


# Initialisation et wrapping des modèles Langchain pour Ragas
llm_model = ChatMistralAI()
embedding_model = MistralAIEmbeddings()

ragas_llm = LangchainLLMWrapper(llm_model)
ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)

evaluation_data = {
    "user_input": questions,
    "response": answers,
    "retrieved_contexts": placeholder_contexts,
    "reference": ground_truths
}

evaluation_dataset = Dataset.from_dict(evaluation_data)

for row in evaluation_dataset:
    result = evaluate_rag(row, rag=rag, llm=ragas_llm, embeddings=ragas_embeddings)
    print(f"Question: {result['user_input']}")
    print(f"Réponse du modèle: {result['model_response']}")
    print(f"Score de fidélité: {result['faithfullness_score']}, Raison: {result['faithfullness_reason']}")
    print(f"Score de pertinence de la réponse: {result['answer_relevancy_score']}, Raison: {result['answer_relevancy_reason']}")
    print(f"Score de précision du contexte: {result['context_precision_score']}, Raison: {result['context_precision_reason']}")
    print(f"Score de rappel du contexte: {result['context_recall_score']}, Raison: {result['context_recall_reason']}")
    print(f"Documents récupérés (extraits): {[doc[:100] + '...' if len(doc) > 100 else doc for doc in result['retrieved_documents']]}")
    print('------------------')
