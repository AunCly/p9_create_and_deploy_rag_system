import os

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

from api.main import app

load_dotenv()

client = TestClient(app)

@pytest.fixture
def get_random_prompt():
    questions = [
        'A quelle date peut-on voir Dany-Io le sculpteur de pierre ?',
        'Nous sommes le 3 septembre, quel est le prochain concert à venir à la Sirène ?',
        'A quelle heure commence le spectacle Premier regard sur L’homme qui danse ?',
        'Quels sont les évènements gratuits en Septembre 2027 ?',
        'Quand se passe le festival de la BD d\'Angoulême en 2026 ?',
    ]

    return questions[os.urandom(1)[0] % len(questions)]

class TestApi:
    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"message": "Alive !"}

    def test_ask_question_without_authorization(self):
        response = client.post("/ask", headers={"x-api-key": "foo"})
        assert response.status_code == 403
        assert response.json() == {"detail": "Forbidden"}

    def test_ask(self, get_random_prompt):
        response = client.post("/ask", json={"prompt": get_random_prompt}, headers={"x-api-key": os.getenv('API_AUTH')})

        response_keys_needed = { "query", "answer", "documents" }

        assert response.status_code == 200
        assert response_keys_needed <= response.json().keys()
        assert response_keys_needed == response.json().keys()

    def test_ask_with_missing_prompt(self):

        response = client.post("/ask", headers={"x-api-key": os.getenv('API_AUTH')})

        assert response.status_code == 422


    def test_rebuild(self):
        response = client.post("/rebuild", headers={"x-api-key": os.getenv('API_AUTH')})

        assert response.status_code == 200
        assert response.json() == {"message": "Index FAISS reconstruit avec succès !"}