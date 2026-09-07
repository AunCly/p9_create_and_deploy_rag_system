import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from script.rag import EventsRag as Rag
from api.prompt_model import PromptModel
from script.create_faiss_index import create_faiss_index

load_dotenv()

api_key_header = APIKeyHeader(name="x-api-key")

@asynccontextmanager
async def lifespan(app: FastAPI):
    base_path = Path(__file__).parent
    if not (base_path / "data" / "raw" / "faiss_langchain_index" / "index.faiss").exists():
        create_faiss_index()
    yield

app = FastAPI(
    lifespan=lifespan,
    title="Api à propos des évènements passés ou à venir en Nouvelle-Aquitaine",
    description="Cette API permet de poser des questions à propos des évènement passé ou à venir en Nouvelle-Aquitaine.",
    version="0.0.1",
    contact={
        "email": "aurelien.clugery@gmail.com",
    }
)

async def verify_api_key(api_key: str = Security(api_key_header)):

    key = os.getenv('API_AUTH')

    if not key or api_key != key:
        raise HTTPException(status_code=403)

    return api_key

@app.get(
    "/health",
    summary="Vérifier la santé de l'API",
    responses={
        200: {"description": "Le serveur de l'API fonctionne correctement"},
    }
)
def health_endpoint():
    return {"message": "Alive !"}

@app.post(
    "/ask",
    dependencies=[Depends(verify_api_key)],
    summary="Poser une question à propos des évènements passés ou à venir en Nouvelle-Aquitaine",
    description="Cette endpoint permet de poser une question à propos des évènements passés ou à venir en Nouvelle-Aquitaine.",
    responses={
        200: {"description": "Réponse à la question posée"},
        401: {"description": "Clé API manquante ou invalide"},
        422: {"description": "Données d'employé invalides (erreur de validation Pydantic)"}
    }
)
def ask_endpoint(prompt: PromptModel):

    tmp = prompt.model_dump()

    rag = Rag()
    response, documents = rag.answer(tmp.get("prompt"))

    clean_documents = []
    for doc in documents:
        # On extrait de manière sécurisée le contenu texte (évite l'erreur numpy)
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        clean_documents.append({"page_content": content})

    return {
        "query": tmp.get("prompt"),
        "answer": response.content,
        "documents": clean_documents
    }

@app.post(
    "/rebuild",
    dependencies=[Depends(verify_api_key)],
    summary="Reconstruire l'index FAISS pour les évènements",
    description="Cette endpoint permet de reconstruire l'index FAISS pour les évènements passés ou à venir en Nouvelle-Aquitaine.",
    responses={
        200: {"description": "Index FAISS reconstruit avec succès"},
        401: {"description": "Clé API manquante ou invalide"},
        422: {"description": "Données d'employé invalides (erreur de validation Pydantic)"}
    }
)
def rebuild_endpoint():

    create_faiss_index()

    return {"message": "Index FAISS reconstruit avec succès !"}
