"""
Candidate generation, reworked from Section 14 of the notebook.

The notebook's version called implicit's `als_model.recommend()`, which
needs the full training user-item sparse matrix to filter already-seen
items. We don't ship that matrix as an artifact (it's large and fully
redundant with `user_seen.pkl`), so instead we score directly via the ALS
factor dot product and filter seen movies ourselves. Mathematically
equivalent for a known user; just doesn't need the extra file.
"""

from typing import Optional

import numpy as np

from .artifacts import ArtifactStore


def generate_candidates(
    store: ArtifactStore,
    user_id: int,
    n_candidates: int = 200,
    seen_movies: Optional[set] = None,
) -> list:
    """Return up to n_candidates (movieId, als_score) pairs for a user,
    excluding anything in seen_movies. Empty list for unknown users --
    the caller falls back to popularity (see recommend.py)."""
    if user_id not in store.user_id_to_idx:
        return []

    if seen_movies is None:
        seen_movies = set()
    elif not isinstance(seen_movies, set):
        seen_movies = set(seen_movies.tolist())
    if not isinstance(seen_movies, set):
        seen_movies = set(seen_movies.tolist())
    user_idx = store.user_id_to_idx[user_id]

    user_vec = store.als_model.user_factors[user_idx]
    all_scores = store.als_model.item_factors @ user_vec  # shape: (n_movies_train,)

    # Partial sort: only need the top (n_candidates + a buffer for filtered-out seen items)
    buffer = min(len(all_scores), n_candidates + len(seen_movies) + 50)
    top_idx = np.argpartition(-all_scores, buffer - 1)[:buffer]
    top_idx = top_idx[np.argsort(-all_scores[top_idx])]

    candidates = []
    for idx in top_idx:
        movie_id = store.idx_to_movie_id[int(idx)]
        if movie_id in seen_movies:
            continue
        candidates.append((movie_id, float(all_scores[idx])))
        if len(candidates) >= n_candidates:
            break

    return candidates
