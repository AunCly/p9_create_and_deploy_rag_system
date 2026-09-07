import os

from dotenv import load_dotenv
from typing_extensions import Annotated, TypedDict
from langsmith import Client, traceable
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from script.rag import EventsRag as Rag

load_dotenv()

client = Client()

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

# Formatage des exemples pour LangSmith
examples = [
    {"inputs": {"question": q}, "outputs": {"answer": a}}
    for q, a in zip(questions, ground_truths)
]

dataset_name = "Events RAG Dataset"
if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(dataset_id=dataset.id, examples=examples)

# ==========================================
# 2. Fonction Cible (Votre RAG)
# ==========================================

rag = Rag()


@traceable()
def predict_rag(inputs: dict) -> dict:
    """Fonction qui sera évaluée. Doit retourner l'answer et les documents."""
    question = inputs["question"]
    answer, documents = rag.answer(question)

    # On s'assure que les documents sont bien des objets avec 'page_content'
    # (requis par les évaluateurs plus bas)
    formatted_docs = []
    if documents:
        for doc in documents:
            if hasattr(doc, 'page_content'):
                formatted_docs.append(doc)
            else:
                formatted_docs.append(Document(page_content=str(doc)))

    return {
        "answer": answer.content if hasattr(answer, 'content') else str(answer),
        "documents": formatted_docs
    }


# ==========================================
# 3. Définition des Évaluateurs (LLM Judges)
# ==========================================

eval_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


# --- A. CORRECTNESS (Exactitude par rapport à la réponse attendue) ---
class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]


correctness_instructions = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. 
(2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the ground truth answer.

A correctness value of True means that the student's answer meets all of the criteria."""

grader_llm = eval_llm.with_structured_output(CorrectnessGrade)


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    answers = f"QUESTION: {inputs['question']}\nGROUND TRUTH ANSWER: {reference_outputs['answer']}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = grader_llm.invoke([
        {"role": "system", "content": correctness_instructions},
        {"role": "user", "content": answers},
    ])
    return grade["correct"]


# --- B. RELEVANCE (Pertinence de la réponse générée) ---
class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "Provide the score on whether the answer addresses the question"]


relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION."""

relevance_llm = eval_llm.with_structured_output(RelevanceGrade)


def relevance(inputs: dict, outputs: dict) -> bool:
    answer = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = relevance_llm.invoke([
        {"role": "system", "content": relevance_instructions},
        {"role": "user", "content": answer},
    ])
    return grade["relevant"]


# --- C. GROUNDEDNESS (Hallucinations - Ragas "Faithfulness") ---
class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[bool, ..., "Provide the score on if the answer hallucinates from the documents"]


grounded_instructions = """You are a teacher grading a quiz. You will be given FACTS and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 
(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS."""

grounded_llm = eval_llm.with_structured_output(GroundedGrade)


def groundedness(inputs: dict, outputs: dict) -> bool:
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    answer = f"FACTS: {doc_string}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = grounded_llm.invoke([
        {"role": "system", "content": grounded_instructions},
        {"role": "user", "content": answer},
    ])
    return grade["grounded"]


# --- D. RETRIEVAL RELEVANCE (Ragas "Context Precision/Recall") ---
class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "True if the retrieved documents are relevant to the question, False otherwise"]


retrieval_relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a set of FACTS provided by the student. Here is the grade criteria to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met."""

retrieval_relevance_llm = eval_llm.with_structured_output(RetrievalRelevanceGrade)


def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    answer = f"FACTS: {doc_string}\nQUESTION: {inputs['question']}"
    grade = retrieval_relevance_llm.invoke([
        {"role": "system", "content": retrieval_relevance_instructions},
        {"role": "user", "content": answer},
    ])
    return grade["relevant"]


# ==========================================
# 4. Exécution de l'Évaluation
# ==========================================

if __name__ == "__main__":
    print("Démarrage de l'évaluation avec LangSmith...")

    experiment_results = client.evaluate(
        predict_rag,
        data=dataset_name,
        evaluators=[correctness, groundedness, relevance, retrieval_relevance],
        experiment_prefix="events-rag-structured-eval",
        metadata={"version": "Gemini Structured Output Evaluators"},
    )

    print("\nÉvaluation terminée ! Consultez les résultats sur https://smith.langchain.com/")
