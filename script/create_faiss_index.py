import json
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import script.utils as utils
import script.downloader as downloader

load_dotenv()

base_path = Path(os.path.dirname(__file__)).parent

def create_faiss_index():
    if not (base_path / "data" / "raw" / "events.json").exists():
        downloader.download_data()

    print("Chargement du JSON...")
    events = pd.read_json(base_path / "data" / "raw" / "events.json")

    columns = {
        'title_fr': 'Titre',
        'description_fr': 'Description',
        'longdescription_fr': 'Description Longue',
        'conditions_fr': 'Conditions',
        'keywords_fr': 'Mots clés',
        'accessibility_label_fr': 'Condition d\'accessibilité',
        'location_city': 'Ville',
        'location_department': 'Département',
    }

    print("Création des documents LangChain...")
    documents = []

    for index, event in events.iterrows():
        text_content = ""

        metadata = {
            "url": event.get('canonicalurl', ''),
            "start_date_begin": event.get('firstdate_begin', ''),
            "start_date_end": event.get('firstdate_end', ''),
            "end_date_begin": event.get('lastdate_begin', ''),
            "end_date_end": event.get('lastdate_end', ''),
            "dates": utils.format_dates(json.loads(event.get('timings', ''))),
            "location_address": event.get('location_address', ''),
            "location_postalcode": event.get('location_postalcode', ''),
            "location_city": event.get('location_city', ''),
            "location_department": event.get('location_department', ''),
            "location_region": event.get('location_region', ''),
            "location_website": event.get('location_website', ''),
            "age_min": event.get('age_min', ''),
            "age_max": event.get('age_max', ''),
            "registration": utils.format_registration(event.get('registration', '')),
        }

        for alias, title in columns.items():
            try:
                val = event.get(alias)
                if pd.isna(val) or val is None:
                    continue
                if alias == 'longdescription_fr':
                    cleaned = utils.clean_html(val)
                    text_content += f'{title}: {cleaned}\n'
                elif alias in ['keywords_fr', 'accessibility_label_fr']:
                    text_content += f'{title}: {" - ".join(val)}\n'
                else:
                    text_content += f'{title}: {val}\n'

            except Exception as ex:
                print(f'Error processing event {index} column {alias}: {ex}')

        doc = Document(page_content=text_content, metadata=metadata)
        documents.append(doc)

    print(f"✅ Création de {len(documents)} documents LangChain terminée !")

    print("Découpage des documents en chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunked_documents = text_splitter.split_documents(documents)

    print("Connexion à l'API Google Generative AI...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
    )

    batch_size = 500
    vector_store = None

    print(f"Début de la vectorisation de {len(chunked_documents)} événements (par lots de {batch_size})...")

    for i in range(0, len(chunked_documents), batch_size):
        batch_docs = chunked_documents[i: i + batch_size]
        print(f"Vectorisation du lot {i} à {i + len(batch_docs)}...")

        if vector_store is None:
            vector_store = FAISS.from_documents(batch_docs, embeddings)
        else:
            vector_store.add_documents(batch_docs)

        if i >= 1000:
            print("Limite de 1000 documents atteinte pour le test. Arrêt de la vectorisation.")
            break

    print("Sauvegarde de la base de données FAISS sur le disque...")
    vector_store.save_local(str(base_path / "data" / "raw" / "faiss_langchain_index"))

    print("✅ Terminé avec succès !")


create_faiss_index()