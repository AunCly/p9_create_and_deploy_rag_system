import asyncio

import pandas as pd
import google.generativeai as genai

from datasets import Dataset
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from script.rag import EventsRag as Rag
from ragas.llms import LangchainLLMWrapper, llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper, embedding_factory
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

# 1. Initialisation du RAG
rag = Rag()

answers = []
placeholder_contexts = []

# 2. Préparation des données (Extraction robuste des chaînes de texte)
for question in questions:
    answer, documents = rag.answer(question)
    answers.append(answer.content if hasattr(answer, 'content') else str(answer))

    # Extraction propre pour obtenir une liste de STRINGS (évite l'erreur PyArrow)
    text_contexts = []
    for doc in documents:
        if hasattr(doc, 'page_content'):
            text_contexts.append(doc.page_content)
        elif isinstance(doc, dict):
            text_contexts.append(doc.get('page_content') or doc.get('content', str(doc)))
        else:
            text_contexts.append(str(doc))

    placeholder_contexts.append(text_contexts)

# Dataset formaté avec la nomenclature Ragas 0.2.x
evaluation_data = {
    "user_input": questions,
    "response": answers,
    "retrieved_contexts": placeholder_contexts,
    "reference": ground_truths
}

evaluation_dataset = Dataset.from_dict(evaluation_data)

# 3. Initialisation des modèles Langchain
client = genai.GenerativeModel("gemini-3.5-flash-lite")
ragas_llm = llm_factory(
    client=client,
    provider="google",
    model="gemini-3.5-flash-lite"
)
ragas_embeddings = embedding_factory(
    model="models/embedding-001"
)


# 4. Fonction d'évaluation avec le décorateur @experiment
@experiment()
async def evaluate_rag(row, llm, embeddings):
    # Nous utilisons directement les données de 'row' préparées en amont.
    # Plus besoin de faire 'rag.query()' ici !

    # Ragas 0.2.x préfère parfois recevoir un dictionnaire ou un objet SingleTurnSample.
    # Nous préparons le dictionnaire exact attendu par les métriques.
    sample = {
        "user_input": row["user_input"],
        "response": row["response"],
        "retrieved_contexts": row["retrieved_contexts"],
        "reference": row["reference"]
    }

    # Evaluate Faithfulness
    faithfullness_scorer = Faithfulness(llm=llm)
    faithfullness_result = faithfullness_scorer.score(sample)

    # Evaluate Answer Relevancy
    answer_relevancy_scorer = AnswerRelevancy(llm=llm, embeddings=embeddings)
    answer_relevancy_result = answer_relevancy_scorer.score(sample)

    # Evaluate Context Precision (Nécessite impérativement 'reference')
    context_precision_scorer = ContextPrecision(llm=llm, embeddings=embeddings)
    context_precision_result = context_precision_scorer.score(sample)

    # Evaluate Context Recall (Nécessite impérativement 'reference')
    context_recall_scorer = ContextRecall(llm=llm, embeddings=embeddings)
    context_recall_result = context_recall_scorer.score(sample)

    # On extrait les valeurs et raisons de manière sécurisée
    # (Selon la version exacte, score() retourne un objet avec .value/.reason ou directement la valeur)
    return {
        **row,
        "faithfullness_score": getattr(faithfullness_result, 'value', faithfullness_result),
        "faithfullness_reason": getattr(faithfullness_result, 'reason', None),
        "answer_relevancy_score": getattr(answer_relevancy_result, 'value', answer_relevancy_result),
        "answer_relevancy_reason": getattr(answer_relevancy_result, 'reason', None),
        "context_precision_score": getattr(context_precision_result, 'value', context_precision_result),
        "context_precision_reason": getattr(context_precision_result, 'reason', None),
        "context_recall_score": getattr(context_recall_result, 'value', context_recall_result),
        "context_recall_reason": getattr(context_recall_result, 'reason', None),
    }


# 5. Exécution de l'évaluation
async def main():
    for row in evaluation_dataset:
        # Il faut ajouter 'await' devant l'appel de la fonction
        result = await evaluate_rag(row, llm=ragas_llm, embeddings=ragas_embeddings)

        print(f"Question: {result['user_input']}")
        print(f"Réponse du modèle: {result['response']}")
        print(f"Score de fidélité: {result['faithfullness_score']}")
        print(f"Raison (fidélité): {result['faithfullness_reason']}")
        print(f"Score de pertinence: {result['answer_relevancy_score']}")
        print(f"Score de précision du contexte: {result['context_precision_score']}")
        print(f"Score de rappel du contexte: {result['context_recall_score']}")

        # Raccourcir le contexte pour l'affichage console
        contexts_preview = [doc[:100] + '...' if len(doc) > 100 else doc for doc in result['retrieved_contexts']]
        print(f"Documents récupérés (extraits): {contexts_preview}")
        print('-' * 40)


# Lancer la boucle asynchrone principale
if __name__ == "__main__":
    asyncio.run(main())