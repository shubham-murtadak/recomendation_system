import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """Module-level client fixture so models are loaded only once."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_redirect(client):
    """Verify root / redirects to /docs Swagger UI."""
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (307, 302, 308)
    assert response.headers["location"] == "/docs"


def test_health_endpoint(client):
    """Verify /health endpoint returns HTTP 200 and system metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "v4" in data["active_models"]
    assert data["total_catalog_items"] > 0


def test_popular_endpoint(client):
    """Verify /popular endpoint returns top items."""
    response = client.get("/popular?limit=5")
    assert response.status_code == 200
    items = response.json()
    assert len(items) == 5
    assert "item_id" in items[0]
    assert "popularity_score" in items[0]


def test_recommendation_v4_personalized(client):
    """Verify /recommend endpoint returns personalized items for known user."""
    response = client.get("/recommend/172?k=5&model_version=v4")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 172
    assert data["model_version"] == "v4"
    assert len(data["recommendations"]) <= 5
    assert data["latency_ms"] > 0
