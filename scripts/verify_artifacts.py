"""
Run this first, right after copying your downloaded Kaggle artifacts into
artifacts/. It checks that every file loads and that recommend_movies()
works for both a known user and an unknown (cold-start) user, before you
move on to building the API around it.

Usage (from the repo root):
    python scripts/verify_artifacts.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from recommender import get_store, recommend_movies  # noqa: E402


def main():
    print("Loading artifacts...")
    store = get_store("artifacts")
    print(f"  ALS user factors shape: {store.als_model.user_factors.shape}")
    print(f"  ALS item factors shape: {store.als_model.item_factors.shape}")
    print(f"  Movies loaded: {len(store.movies):,}")
    print(f"  Users with training history: {len(store.user_seen):,}")

    known_user_id = next(iter(store.user_id_to_idx))
    print(f"\nRecommending for known user {known_user_id}...")
    recs = recommend_movies(store, known_user_id, top_k=5)
    print(recs.to_string(index=False))
    assert (recs["source"] == "two_stage_model").all(), "Expected two_stage_model source for a known user"

    fake_user_id = max(store.user_id_to_idx.keys()) + 999999
    print(f"\nRecommending for unknown user {fake_user_id} (should fall back to popularity)...")
    recs_cold = recommend_movies(store, fake_user_id, top_k=5)
    print(recs_cold.to_string(index=False))
    assert (recs_cold["source"] == "popularity_fallback").all(), "Expected popularity_fallback source"

    print("\nAll checks passed -- ready to build the API around this.")


if __name__ == "__main__":
    main()
