import json
import os
from pathlib import Path
import requests

region = 'Nouvelle-Aquitaine'
date = 2026

# Remplacement de /records par /exports/json et suppression de la limite
url = f'https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/exports/json?lang=fr&refine=location_region%3A{region}&refine=firstdate_begin%3A{date}&refine=firstdate_end%3A{date}&refine=lastdate_begin%3A{date}&refine=lastdate_end%3A{date}'

headers = {
    "Content-Type": "application/json"
}

print(f"Téléchargement des données depuis l'API d'export (cela peut prendre quelques secondes)...")

response = requests.get(url, headers=headers)

if response.status_code == 200:
    # L'API d'export renvoie directement une liste d'objets JSON
    data = response.json()
    print(f"Succès : {len(data)} événements récupérés au total.")

    base_path = Path(os.path.dirname(__file__)).parent
    output_dir = base_path / 'data' / 'raw'

    # Sécurité : on s'assure que les dossiers existent avant d'écrire
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'events.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        # On sauvegarde le JSON proprement avec une indentation pour qu'il soit lisible
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Fichier sauvegardé avec succès dans : {output_file}")

else:
    print(f"Échec de la requête : Erreur {response.status_code}")
    print(response.text)