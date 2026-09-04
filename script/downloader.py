import json
import os
from pathlib import Path

import requests


def download_data():
    ville = 'La+Rochelle'
    date = 2026

    url = f'https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/exports/json?lang=fr&refine=location_city%3A{ville}&refine=firstdate_begin%3A{date}&refine=firstdate_end%3A{date}&refine=lastdate_begin%3A{date}&refine=lastdate_end%3A{date}'

    print(f"Téléchargement des données depuis l'API d'export (cela peut prendre quelques secondes)...")

    response = requests.get(url, headers={
        "Content-Type": "application/json"
    })

    if response.status_code == 200:
        print("Début de la réponse brute :", response.text[:200])

        data = response.json()
        print(f"Succès : {len(data)} événements récupérés au total.")

        base_path = Path(os.path.dirname(__file__)).parent
        output_dir = base_path / 'data' / 'raw'

        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / 'events.json'

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Fichier sauvegardé avec succès dans : {output_file}")

    else:
        print(f"Échec de la requête : Erreur {response.status_code}")
        print(response.text)


download_data()