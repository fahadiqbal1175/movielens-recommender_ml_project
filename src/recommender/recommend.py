"""
The public entry point: recommend_movies(). Ported from Section 17 (final
two-stage system) and Section 22 (cold-start handling) of the notebook.
"""

from typing import Optional

import pandas as pd

from .artifacts import ArtifactStore
from .candidates import generate_candidates
from .features import build_feature_row


def _popularity_fallback(store: ArtifactStore, top_k: int, seen_movies: set) -> pd.DataFrame:
    if seen_movies is None:
        seen_movies = set()
    elif not isinstance(seen_movies, set):
        seen_movies = set(seen_movies.tolist())
    ranked_ids = [m for m in store.popularity_ranking if m not in seen_movies][:top_k]
    result = store.movies[store.movies["movieId"].isin(ranked_ids)][["movieId", "title", "genres"]].copy()
    # preserve popularity order rather than whatever isin() returns
    order = {m: i for i, m in enumerate(ranked_ids)}
    result["_order"] = result["movieId"].map(order)
    result = result.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    result["score"] = None
    result["source"] = "popularity_fallback"
    return result


def recommend_movies(
    store: ArtifactStore,
    user_id: int,
    top_k: int = 10,
    n_candidates: Optional[int] = None,
) -> pd.DataFrame:
    """Return top_k (movieId, title, genres, score, source) recommendations
    for user_id. Falls back to popularity for unknown/cold-start users."""
    n_candidates = n_candidates or store.config.get("n_candidates_default", 200)
    seen_movies = store.user_seen.get(user_id)

    candidates = generate_candidates(store, user_id, n_candidates=n_candidates, seen_movies=seen_movies)
    if not candidates:
        return _popularity_fallback(store, top_k, seen_movies)

    feature_rows = [build_feature_row(store, user_id, movie_id) for movie_id, _ in candidates]
    X = pd.DataFrame(feature_rows)[store.feature_cols()].values
    scores = store.ranking_model.predict_proba(X)[:, 1]

    candidate_ids = [m for m, _ in candidates]
    ranked = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)[:top_k]
    ranked_ids = [m for m, _ in ranked]
    score_map = dict(ranked)

    result = store.movies[store.movies["movieId"].isin(ranked_ids)][["movieId", "title", "genres"]].copy()
    result["score"] = result["movieId"].map(score_map)
    result["source"] = "two_stage_model"
    result = result.sort_values("score", ascending=False).reset_index(drop=True)
    return result.head(top_k)
