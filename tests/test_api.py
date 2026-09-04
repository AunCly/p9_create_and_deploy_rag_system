import os
from pathlib import Path
from unittest.mock import patch

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from main import app
import pandas as pd

load_dotenv()

client = TestClient(app)

@pytest.fixture
def load_employees_from_raw_data():
    base_dir = Path(__file__).resolve().parent.parent

    data_sirh = pd.read_csv(base_dir / 'data/raw/extrait_sirh.csv', sep=',', na_values=[''], quotechar='"')
    data_eval = pd.read_csv(base_dir / 'data/raw/extrait_eval.csv', sep=',', na_values=[''], quotechar='"')
    data_sondage = pd.read_csv(base_dir / 'data/raw/extrait_sondage.csv', sep=',', na_values=[''], quotechar='"')

    data_eval['id_employee'] = data_eval['eval_number'].apply(lambda x : x.split('E_')[1])
    data_eval['id_employee'] = data_eval['id_employee'].astype(int)

    data_sondage['id_employee'] = data_sondage['code_sondage']
    data_sondage['id_employee'] = data_sondage['id_employee'].astype(int)

    # Fusion des fichiers
    data = data_sirh.merge(data_eval, how='inner')
    data = data.merge(data_sondage, how='inner')

    return data

@pytest.fixture
def get_random_employee(load_employees_from_raw_data):
    all_employees = load_employees_from_raw_data
    sample_df = all_employees.sample(1)
    data_dict = sample_df.to_dict(orient='records')[0]
    return data_dict

@pytest.fixture
def get_random_employees(load_employees_from_raw_data):
    all_employees = load_employees_from_raw_data
    return all_employees.sample(10)

class TestApi:
    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"message": "Alive !"}

    def test_predict_with_wrong_api_key(self):
        response = client.post("/predict", headers={"x-api-key": "foo"})
        assert response.status_code == 403
        assert response.json() == {"detail": "Forbidden"}

    def test_predict(self, get_random_employee):
        response = client.post("/predict", json=get_random_employee, headers={"x-api-key": os.getenv('API_KEY')})

        response_keys_needed = {"probability", "prediction", "employee_id"}

        assert response.status_code == 200
        assert response_keys_needed <= response.json().keys()
        assert response_keys_needed == response.json().keys()

    def test_predict_with_missing_fields(self, get_random_employee):
        random_employee = get_random_employee

        del(random_employee['genre'])

        response = client.post("/predict", json=get_random_employee, headers={"x-api-key": os.getenv('API_KEY')})

        assert response.status_code == 422

    def test_predict_batch(self, get_random_employees):
        response = client.post("/predict/batch", json=get_random_employees.to_dict(orient='records'), headers={"x-api-key": os.getenv('API_KEY')})

        response_keys_needed = {"probability", "prediction", "employee_id"}

        assert response.status_code == 200
        results = response.json()
        assert isinstance(results, list)
        assert len(results) == 10
        for result in results:
            assert response_keys_needed == result.keys()


    def test_predict_batch_with_missing_fields(self, get_random_employees):

        random_employees = get_random_employees

        del(random_employees['genre'])

        response = client.post("/predict/batch", json=random_employees.to_dict(orient='records'), headers={"x-api-key": os.getenv('API_KEY')})

        assert response.status_code == 422

    def test_history(self):
        with patch('database.database_manager.get_predictions', return_value=[]):
            response = client.get("/predict/history", headers={"x-api-key": os.getenv('API_KEY')})
            assert response.status_code == 200
            assert isinstance(response.json(), list)

    def test_history_with_wrong_api_key(self):
        response = client.get("/predict/history", headers={"x-api-key": "wrong"})
        assert response.status_code == 403

    def test_lifespan_non_production(self):
        with patch('database.database_manager.create_database') as mock_create:
            with patch('database.seeding.seed') as mock_seed:
                with patch.dict(os.environ, {'ENVIRONMENT': 'development'}):
                    with TestClient(app):
                        pass
                mock_create.assert_called_once()
                mock_seed.assert_called_once()
