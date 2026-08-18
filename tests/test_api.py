"""
Tests for the FastAPI service (app/main.py). Like tests/test_recommender.py,
these run against the real trained artifacts rather than mocks -- requires
artifacts/ to be populated (see README).

Run with: pytest tests/
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from app.main import app  # noqa: E402
from recommender import get_store  # noqa: E402

ARTIFACT_DIR = str(REPO_ROOT / "artifacts")


@pytest.fixture(scope="module")
def client():
    # TestClient as a context manager triggers the app's lifespan handler,
    # so artifacts load once for the whole module, same as the `store`
    # fixture in test_recommender.py.
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def known_user_id():
    store = get_store(ARTIFACT_DIR)
    return next(iter(store.user_id_to_idx))


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["artifacts_loaded"] is True
    assert body["n_users"] > 0
    assert body["n_movies"] > 0


def test_recommend_known_user_uses_two_stage_model(client, known_user_id):
    resp = client.get(f"/recommend/{known_user_id}", params={"top_k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == known_user_id
    assert body["source"] == "two_stage_model"
    assert len(body["recommendations"]) <= 5
    assert all(r["source"] == "two_stage_model" for r in body["recommendations"])
    assert all(r["score"] is not None for r in body["recommendations"])


def test_recommend_unknown_user_falls_back_to_popularity(client, known_user_id):
    fake_user_id = known_user_id + 9_000_000
    resp = client.get(f"/recommend/{fake_user_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "popularity_fallback"
    assert all(r["source"] == "popularity_fallback" for r in body["recommendations"])
    assert all(r["score"] is None for r in body["recommendations"])


def test_default_top_k_is_ten(client, known_user_id):
    resp = client.get(f"/recommend/{known_user_id}")
    assert resp.status_code == 200
    assert resp.json()["top_k"] == 10


def test_top_k_out_of_range_rejected(client, known_user_id):
    assert client.get(f"/recommend/{known_user_id}", params={"top_k": 0}).status_code == 422
    assert client.get(f"/recommend/{known_user_id}", params={"top_k": 1000}).status_code == 422


def test_non_integer_user_id_rejected(client):
    assert client.get("/recommend/not-a-number").status_code == 422


def test_non_positive_user_id_rejected(client):
    assert client.get("/recommend/0").status_code == 422
    assert client.get("/recommend/-5").status_code == 422
