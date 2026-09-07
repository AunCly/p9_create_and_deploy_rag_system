# Documentation technique — Puls-Events RAG

POC d'un chatbot RAG (Retrieval-Augmented Generation) répondant à des questions sur les événements culturels à venir en Nouvelle-Aquitaine (La Rochelle, données 2026), exposé via une API FastAPI.

## Sommaire

- [Architecture](#architecture)
- [Stack technique](#stack-technique)
- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Configuration (variables d'environnement)](#configuration-variables-denvironnement)
- [Pipeline de données](#pipeline-de-données)
- [Cœur du RAG](#cœur-du-rag)
- [API](#api)
- [Tests](#tests)
- [Évaluation](#évaluation)
- [Déploiement (Docker)](#déploiement-docker)
- [Limites connues](#limites-connues)

## Architecture

```
                       ┌─────────────────────────┐
                       │ OpenDataSoft (OpenAgenda)│
                       │  API évènements publics   │
                       └────────────┬─────────────┘
                                    │ downloader.py
                                    ▼
                        data/raw/events.json
                                    │ create_faiss_index.py
                                    │  - nettoyage / mise en forme (utils.py)
                                    │  - chunking (RecursiveCharacterTextSplitter)
                                    │  - embeddings (Google gemini-embedding-001)
                                    ▼
                data/raw/faiss_langchain_index/ (index FAISS local)
                                    │
                                    ▼
   Client ──HTTP──▶ FastAPI (api/main.py) ──▶ EventsRag (script/rag.py)
                          │  auth: x-api-key         │  1. similarity_search_with_score (top-k)
                          │                          │  2. construction du prompt (contexte + garde-fous)
                          │                          │  3. appel LLM (Google gemini-3.5-flash-lite)
                          ▼                          ▼
                    JSON {query, answer, documents}
```

Le système suit un pipeline RAG classique : ingestion → vectorisation → indexation FAISS → récupération sémantique → génération augmentée par contexte via un LLM.

## Stack technique

| Domaine | Technologie |
|---|---|
| API | FastAPI (+ Uvicorn/`fastapi run`) |
| Orchestration LLM | LangChain (`langchain`, `langchain-community`, `langchain-google-vertexai`) |
| Embeddings | Google Generative AI — `gemini-embedding-001` (`langchain-google-genai`) |
| LLM de génération | Google Gemini — `gemini-3.5-flash-lite` via `init_chat_model` |
| Vector store | FAISS (`faiss-cpu`), persisté sur disque |
| Données sources | API OpenDataSoft / OpenAgenda (`requests`) |
| Manipulation de données | pandas |
| Tests | pytest, `fastapi.testclient` |
| Évaluation | ragas, LangSmith (`langsmith`) |
| Gestion de dépendances | `uv` (pyproject.toml + uv.lock) |
| Conteneurisation | Docker, docker-compose |
| Autre dépendance présente | `mistralai` (client Mistral, utilisé historiquement pour les embeddings — voir [Limites connues](#limites-connues)) |

## Structure du projet

```
.
├── api/
│   ├── main.py            # Application FastAPI (endpoints, auth, lifespan)
│   └── prompt_model.py    # Schéma Pydantic de la requête /ask
├── script/
│   ├── config.py          # Chargement de la config (.env)
│   ├── downloader.py       # Téléchargement des données OpenAgenda -> data/raw/events.json
│   ├── embedding.py        # Client Mistral pour générer un embedding (legacy)
│   ├── create_faiss_index.py  # Pipeline complet : JSON -> documents -> chunks -> index FAISS
│   ├── rag.py              # Classe EventsRag : recherche + génération de réponse
│   ├── utils.py            # Nettoyage HTML, formatage des dates et des inscriptions
│   ├── chat.py             # Script manuel d'appel au RAG (debug/démo en CLI)
│   ├── evaluate_index.py   # Script d'inspection de l'index FAISS
│   ├── evaluate_rag.py     # Évaluation du RAG via LangSmith (LLM-judges)
│   └── datasets/hf_doc_qa_eval.csv
├── tests/
│   ├── test_api.py         # Tests des endpoints FastAPI
│   ├── test_donwloader.py  # Test du téléchargement des données
│   └── test_rag.py         # Tests d'initialisation, de réponse et de reconstruction d'index
├── data/raw/                # events.json + index FAISS (généré, non versionné)
├── notebooks/exploration.ipynb
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml / uv.lock
└── main.py                  # Point d'entrée générique (non utilisé par l'API)
```

## Installation

Prérequis : Python ≥ 3.13, [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Créer un fichier `.env` à la racine (voir la section suivante pour les clés requises).

Lancer l'API en local :

```bash
uv run fastapi dev api/main.py
```

## Configuration (variables d'environnement)

| Variable | Usage |
|---|---|
| `GOOGLE_API_KEY` | Authentification aux API Google Generative AI (embeddings `gemini-embedding-001` et LLM `gemini-3.5-flash-lite`) |
| `API_AUTH` | Clé API attendue dans le header `x-api-key` pour accéder à `/ask` et `/rebuild` |
| `MISTRAL_API_KEY` | Clé pour le client Mistral (`script/embedding.py`, legacy, non utilisé par le pipeline courant) |
| `LANGSMITH_API_KEY` | Authentification LangSmith pour `script/evaluate_rag.py` |

Le chargement se fait via `python-dotenv` (`load_dotenv()`) dans chaque module qui en a besoin.

## Pipeline de données

### 1. Téléchargement — `script/downloader.py`

Interroge l'API publique OpenDataSoft (jeu de données `evenements-publics-openagenda`) filtrée sur la ville de La Rochelle et l'année 2026, et sauvegarde le résultat brut dans `data/raw/events.json`. Déclenché automatiquement si ce fichier est absent lors de la création de l'index.

### 2. Construction de l'index — `script/create_faiss_index.py`

Pour chaque événement du JSON :
- extraction et concaténation des champs pertinents (titre, description, description longue, conditions, mots-clés, accessibilité, ville, département) ;
- nettoyage HTML des champs longs (`utils.clean_html`) ;
- formatage des dates d'ouverture (`utils.format_dates`) et des modalités d'inscription (`utils.format_registration`) en métadonnées ;
- découpage en chunks via `RecursiveCharacterTextSplitter` (taille 1000, overlap 150) ;
- vectorisation par lots de 500 documents avec `GoogleGenerativeAIEmbeddings` (`gemini-embedding-001`) ;
- sauvegarde de l'index FAISS dans `data/raw/faiss_langchain_index/`.

> ⚠️ Le script contient une limite codée en dur : la vectorisation s'arrête après le premier lot dont l'index de départ dépasse 1000 (`if i >= 1000: break`), ce qui plafonne le corpus indexé à ~1500 chunks. À lever si l'objectif est de couvrir l'intégralité du jeu de données.

### 3. Inspection — `script/evaluate_index.py`

Script utilitaire chargeant l'index FAISS existant pour en afficher les statistiques (nombre de vecteurs, dimension) et tester une recherche de similarité manuelle.

## Cœur du RAG

Classe `EventsRag` (`script/rag.py`) :

1. **Initialisation** : charge les embeddings Google, le modèle de chat (`init_chat_model("google_genai:gemini-3.5-flash-lite", max_retries=10, timeout=120)`) et l'index FAISS local (`load_faiss`, échoue explicitement si l'index n'existe pas sur disque).
2. **`search_documents(prompt, k=5)`** : recherche par similarité vectorielle (`similarity_search_with_score`), retourne les `k` documents les plus proches avec leur score.
3. **`construct_prompt(prompt, retrieved_docs)`** : construit un `SystemMessage` contenant les garde-fous métier (répond uniquement sur les événements 2026 à La Rochelle ; refuse toute autre ville/année ; pas de suppositions hors documents fournis) suivi du contenu des documents récupérés (métadonnées + texte), puis un `HumanMessage` avec la question de l'utilisateur.
4. **`answer(prompt)`** : orchestre recherche + construction du prompt + appel au LLM, retourne `(response, documents)`.

## API

Définie dans `api/main.py`. Documentation interactive générée automatiquement par FastAPI sur `/docs` (Swagger) et `/redoc`.

### Authentification

Toutes les routes sauf `/health` requièrent un header `x-api-key` correspondant à la variable d'environnement `API_AUTH`. Sinon : `403 Forbidden`.

### Cycle de vie

Au démarrage (`lifespan`), l'API vérifie la présence de l'index FAISS (`data/raw/faiss_langchain_index/index.faiss`) et le (re)construit automatiquement via `create_faiss_index()` s'il est absent.

### Endpoints

| Méthode | Route | Auth | Description |
|---|---|---|---|
| GET | `/health` | non | Vérifie que l'API répond (`{"message": "Alive !"}`) |
| POST | `/ask` | oui | Body `{"prompt": "<question>"}` → `{"query", "answer", "documents"}` (documents = contenu texte des chunks utilisés comme contexte) |
| POST | `/rebuild` | oui | Reconstruit intégralement l'index FAISS (relance tout le pipeline d'ingestion) |

Codes d'erreur : `401`/`403` (clé API invalide/absente), `422` (payload invalide, ex. `prompt` manquant sur `/ask`).

## Tests

Exécution :

```bash
uv run pytest
```

| Fichier | Couverture |
|---|---|
| `tests/test_api.py` | `/health`, `/ask` (avec/sans auth, prompt manquant), `/rebuild` — via `fastapi.testclient.TestClient` |
| `tests/test_donwloader.py` | Téléchargement effectif des données OpenAgenda |
| `tests/test_rag.py` | Initialisation de `EventsRag`, génération d'une réponse, reconstruction de l'index |

⚠️ Ces tests appellent réellement les API externes (OpenDataSoft, Google Generative AI) — ce sont des tests d'intégration nécessitant un `.env` valide et une connexion réseau, pas des tests unitaires isolés/mockés. `test_rebuild` et `test_build_index` reconstruisent l'index complet à chaque exécution (coût API + temps).

## Évaluation

Deux approches disponibles dans `script/` :

- **`evaluate_rag.py`** : évaluation end-to-end via **LangSmith**. Un jeu de 5 questions/réponses de référence est enregistré comme dataset LangSmith, puis le pipeline `predict_rag` est évalué par 4 LLM-judges (Gemini) :
  - `correctness` — exactitude factuelle vs réponse de référence ;
  - `relevance` — pertinence de la réponse par rapport à la question ;
  - `groundedness` — absence d'hallucination par rapport aux documents récupérés ;
  - `retrieval_relevance` — pertinence des documents récupérés par rapport à la question.

  Résultats consultables sur `smith.langchain.com`. Nécessite `LANGSMITH_API_KEY`.

## Déploiement (Docker)

```bash
docker compose up --build
```

- Image basée sur `python:3.13-slim`, dépendances installées via `uv sync --frozen`.
- L'API est servie en interne sur le port `7860` (`fastapi run api/main.py --port 7860`), exposée sur l'hôte via le port `8000` (mapping `8000:7860` dans `docker-compose.yml`).
- Les dossiers `./api` et `./data` sont montés en volumes.
- Le fichier `.env` n'est pas copié dans l'image (absent du `Dockerfile`/`docker-compose.yml`) : les variables d'environnement doivent être injectées séparément (ex. `env_file: .env` à ajouter au compose) pour un déploiement fonctionnel.

## Limites connues

- **Plafond de vectorisation** : `create_faiss_index.py` interrompt l'indexation après ~1000 documents traités (voir [Pipeline de données](#pipeline-de-données)), donc l'index ne couvre pas nécessairement tous les événements téléchargés.
- **`script/chat.py`** suppose une réponse structurée de type `answer.content[0]['text']`, alors que `EventsRag.answer` retourne un `AIMessage` dont `.content` est généralement une chaîne simple avec le modèle Gemini configuré — à vérifier/aligner avec le format réel retourné par `init_chat_model`.
- **Tests non isolés** : les suites de tests appellent les services externes réels (pas de mocks), donc dépendent de la disponibilité des API tierces, de quotas, et engendrent des coûts/latences à chaque run CI.
