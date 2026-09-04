import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
import warnings

from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")

load_dotenv()
base_path = Path(os.path.dirname(__file__)).parent


def load_faiss():

    if not base_path:
        raise ValueError("Base path is not set.")

    if not (base_path / "data" / "raw" / "faiss_langchain_index" / "index.faiss").exists():
        raise FileNotFoundError("Index file not found at the specified path.")

    if not (base_path / "data" / "raw" / "faiss_langchain_index" / "index.pkl").exists():
        raise FileNotFoundError("Documents file not found at the specified path.")

    vectorstore = FAISS.load_local(
        folder_path=str(base_path / "data" / "raw" / "faiss_langchain_index"),
        embeddings=GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
        ),
        allow_dangerous_deserialization=True
    )

    return vectorstore


def construct_prompt(prompt, retrieved_docs):
    system_prompt = ("Vous êtes un assistant qui répond aux questions sur les événements culturels en Nouvelle-Aquitaine."
                     "Les réponses doivent être coutes, concises, précises et basées sur les documents fournis."
                     "Le ton doit être amical et engageant, et les réponses doivent être en français."
                     "Ne faites pas de suppositions et ne fournissez pas d'informations qui ne sont pas présentes dans les documents."
                     "Si la question ne concerne pas les événements passés ou a venir en 2026 à La Rochelle, répondez que vous ne pouvez pas répondre à cette question."
                     "Si les questions portent sur d'autres années, répondez que vous ne pouvez pas répondre à cette question."
                     "Si les questions portent sur d'autres ville que La Rochelle, répondez que vous ne pouvez pas répondre à cette question.")

    if not retrieved_docs:
        system_prompt += "Aucun document pertinent n'a été trouvé pour répondre à la question."
    else:
        for i, (doc_obj, score) in enumerate(retrieved_docs):
            # 1. On transforme le dictionnaire de métadonnées en une chaîne de caractères
            # Exemple : "ville: Bordeaux, date: 2026-09-15, type: Concert"
            if doc_obj.metadata:
                meta_str = " | ".join([f"{key.capitalize()}: {value}" for key, value in doc_obj.metadata.items()])
                en_tete = f"Document {i + 1} [{meta_str}]"
            else:
                en_tete = f"Document {i + 1}"

            # 2. On assemble l'en-tête et le contenu
            system_prompt += f"{en_tete}:\n{doc_obj.page_content}\n\n"

    messages = [
        SystemMessage(system_prompt),
        HumanMessage(prompt)
    ]

    return messages


class EventsRag:

    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
        )
        self.model = init_chat_model(
    "google_genai:gemini-3.5-flash-lite",
            max_retries=10,
            timeout=120,
        )
        self.vectorstore = load_faiss()

    def search_documents(self, prompt, k=5):
        docs_and_scores = self.vectorstore.similarity_search_with_score(prompt, k=k)
        return docs_and_scores

    def embed_text(self, text):
        embedding = self.embeddings.embed_query(text)
        return embedding

    def answer(self, prompt):
        documents = self.search_documents(prompt)
        message = construct_prompt(prompt, documents)
        response = self.model.invoke(message)
        return response, documents






