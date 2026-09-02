import json
from datetime import datetime
import re
import html

import numpy as np
import pandas as pd

def clean_html(text):
    """
    Cleans HTML tags from the given text.

    Args:
        text (str): The input text containing HTML tags.

    Returns:
        str: The cleaned text without HTML tags.
    """
    # 1. Décoder les entités HTML (transforme &nbsp; en espace, &lt; en <, etc.)
    text = html.unescape(text)

    # 2. Supprimer explicitement les résidus de &nbsp; (minuscules, majuscules ou avec point-virgule manquant)
    text = re.sub(r'&nbsp;?', ' ', text, flags=re.IGNORECASE)

    # 3. Supprimer les balises HTML
    text = re.sub(r'<[^>]+>', '', text)

    # 4. Supprimer TOUTES les variations de \n et \r (un ou plusieurs antislashs)
    text = re.sub(r'\\+n', ' ', text)
    text = re.sub(r'\\+r', ' ', text)

    # 5. Supprimer les vrais retours à la ligne et espaces insécables (\xa0)
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\xa0', ' ')

    # 6. Fusionner tous les espaces multiples en un seul
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def format_dates(timings):

    """
    Formats the 'timings' field from the event data into a human-readable string.

    Args:
        timings (list): A list of timing dictionaries.

    Returns:
        str: A formatted string representing the event dates.
    """
    print(f"Timings : {timings}")

    if not timings:
        return ""

    if len(timings) == 1:
        formatted_dates = "Date d'ouverture : "
    else:
        formatted_dates = "Dates d'ouverture : "

    for timing in timings:
        print(timing)
        start_date = datetime.fromisoformat(timing.get('begin', ''))
        end_date = datetime.fromisoformat(timing.get('end', ''))

        if start_date.day == end_date.day and start_date.month == end_date.month and start_date.year == end_date.year:
            date = f"Le {start_date.date()} de {start_date.hour}H{start_date.minute:02d} à {end_date.hour}H{end_date.minute:02d}"
        else:
            date = f"Du {start_date.date()} {start_date.hour}H{start_date.minute:02d} au {end_date.date()} {end_date.hour}H{end_date.minute:02d}"
        formatted_dates += date + "; "

    print(f"Formatted dates: {formatted_dates}")
    return formatted_dates

def format_registration(registrations):
    """
    Formats the 'registration' field from the event data into a human-readable string.
    """
    # Vérification sécurisée pour les valeurs vides, None ou NaN de Pandas
    if registrations is None or pd.isna(registrations) or registrations == "nan":
        return ""

    # Sécurité supplémentaire : si c'est un float (ce qui arrive avec les NaN de pandas)
    if isinstance(registrations, float):
        return ""

    if not isinstance(registrations, str):
        registrations = str(registrations)

    if not registrations.strip():
        return ""

    print(f"Registration : {registrations}")

    try:
        registrations_data = json.loads(registrations)
    except (json.JSONDecodeError, TypeError):
        return ""

    formatted_registrations = ""

    for registration in registrations_data:
        formatted_registrations += "{} : {}; ".format(
            registration.get('type', ''),
            registration.get('value', '')
        )

    return formatted_registrations.strip()

