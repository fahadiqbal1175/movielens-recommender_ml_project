"""
Per-pair feature computation, ported from Section 15/16 of the notebook.

This mirrors the notebook's SCALAR build_feature_row() (not the vectorized
bulk version from Section 16) because at inference time we only ever score
a few hundred candidates for one user at a time -- a Python-level loop over
that many rows is fine, and keeps this module simple and dependency-light.
"""

from .artifacts import ArtifactStore


def als_score(store: ArtifactStore, user_id: int, movie_id: int) -> float:
    if user_id not in store.user_id_to_idx or movie_id not in store.movie_id_to_idx:
        return 0.0
    u = store.user_id_to_idx[user_id]
    m = store.movie_id_to_idx[movie_id]
    return float(store.als_model.user_factors[u] @ store.als_model.item_factors[m])


def genre_overlap(store: ArtifactStore, user_id: int, movie_id: int) -> float:
    user_genres = store.user_top_genres.get(user_id, set())
    movie_genre_list = store.movies_genres.get(movie_id, [])
    if not user_genres or not isinstance(movie_genre_list, list) or not movie_genre_list:
        return 0.0
    overlap = len(user_genres.intersection(movie_genre_list))
    return overlap / max(len(movie_genre_list), 1)


def build_feature_row(store: ArtifactStore, user_id: int, movie_id: int) -> dict:
    global_mean = store.global_mean_rating()

    if user_id in store.user_features.index:
        uf = store.user_features.loc[user_id]
        user_n_ratings = float(uf["user_n_ratings"])
        user_avg_rating = float(uf["user_avg_rating"])
    else:
        user_n_ratings, user_avg_rating = 0.0, global_mean

    if movie_id in store.movie_features.index:
        mf = store.movie_features.loc[movie_id]
        movie_n_ratings = float(mf["movie_n_ratings"])
        movie_bayesian_score = float(mf["bayesian_score"])
    else:
        movie_n_ratings, movie_bayesian_score = 0.0, global_mean

    return {
        "als_score": als_score(store, user_id, movie_id),
        "user_n_ratings": user_n_ratings,
        "user_avg_rating": user_avg_rating,
        "movie_n_ratings": movie_n_ratings,
        "movie_bayesian_score": movie_bayesian_score,
        "genre_overlap": genre_overlap(store, user_id, movie_id),
    }
