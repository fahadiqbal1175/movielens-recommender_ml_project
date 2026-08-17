"""
Basic tests for the recommender package. Requires artifacts/ to be
populated (see README) since these test against the real trained model
rather than mocks -- that's deliberate for a portfolio project: it proves
the actual artifacts work, not just the code shape.

Run with: pytest tests/
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from recommender import get_store, recommend_movies  # noqa: E402

ARTIFACT_DIR = str(Path(__file__).resolve().parent.parent / "artifacts")


@pytest.fixture(scope="module")
def store():
    return get_store(ARTIFACT_DIR)


def test_artifacts_load(store):
    assert store.als_model is not None
    assert store.ranking_model is not None
    assert len(store.movies) > 0


def test_known_user_gets_two_stage_recommendations(store):
    known_user_id = next(iter(store.user_id_to_idx))
    recs = recommend_movies(store, known_user_id, top_k=10)
    assert len(recs) <= 10
    assert (recs["source"] == "two_stage_model").all()
    # must not recommend anything already seen
    seen = store.user_seen.get(known_user_id, set())
    assert not set(recs["movieId"]).intersection(seen)


def test_unknown_user_falls_back_to_popularity(store):
    fake_user_id = max(store.user_id_to_idx.keys()) + 999999
    recs = recommend_movies(store, fake_user_id, top_k=10)
    assert len(recs) <= 10
    assert (recs["source"] == "popularity_fallback").all()


def test_top_k_is_respected(store):
    known_user_id = next(iter(store.user_id_to_idx))
    recs = recommend_movies(store, known_user_id, top_k=3)
    assert len(recs) <= 3
