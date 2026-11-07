from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_report_endpoints_return_generated_reports():
    for path in ("/api/evaluation", "/api/drift", "/api/shap", "/api/mlflow"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.json().get("available") is True


def test_copilot_status_endpoint():
    response = client.get("/api/copilot/status")

    assert response.status_code == 200
    assert response.json()["provider"] in {"none", "ollama", "openai"}
