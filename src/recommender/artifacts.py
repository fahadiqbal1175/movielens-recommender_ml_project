"""
Loads everything exported from Section 24 of the Colab/Kaggle notebook
(als_model.pkl, ranking_model.pkl, id_mappings.pkl, movies.csv,
user_features.csv, movie_features.csv, user_top_genres.pkl, user_seen.pkl,
config.json) into a single in-memory object.

This is the ONLY place that touches the artifact files directly -- every
other module in this package receives an already-loaded ArtifactStore
instead of reading files itself. That keeps file I/O in one place, which
matters once this gets wrapped by a FastAPI app that should load artifacts
exactly once at startup, not on every request.
"""

import json
import os
import pickle
from functools import lru_cache

import pandas as pd


class ArtifactStore:
    def __init__(self, artifact_dir: str):
        self.artifact_dir = artifact_dir
        self._load()

    def _path(self, filename: str) -> str:
        return os.path.join(self.artifact_dir, filename)

    def _load(self):
        missing = [
            f for f in [
                "als_model.pkl", "ranking_model.pkl", "id_mappings.pkl",
                "movies.csv", "user_features.csv", "movie_features.csv",
                "user_top_genres.pkl", "user_seen.pkl", "config.json",
            ]
            if not os.path.exists(self._path(f))
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing artifact files in {self.artifact_dir}: {missing}. "
                "Copy the full 'artifacts/' folder downloaded from your Kaggle "
                "notebook's Output tab into this repo's artifacts/ directory."
            )

        with open(self._path("als_model.pkl"), "rb") as f:
            self.als_model = pickle.load(f)

        with open(self._path("ranking_model.pkl"), "rb") as f:
            self.ranking_model = pickle.load(f)

        with open(self._path("id_mappings.pkl"), "rb") as f:
            mappings = pickle.load(f)
        self.user_id_to_idx = mappings["user_id_to_idx"]
        self.movie_id_to_idx = mappings["movie_id_to_idx"]
        self.idx_to_user_id = mappings["idx_to_user_id"]
        self.idx_to_movie_id = mappings["idx_to_movie_id"]

        self.movies = pd.read_csv(self._path("movies.csv"))
        self.movies_genres = self.movies.set_index("movieId")["genres"].str.split("|")

        self.user_features = pd.read_csv(self._path("user_features.csv"), index_col=0)
        self.movie_features = pd.read_csv(self._path("movie_features.csv"), index_col=0)

        with open(self._path("user_top_genres.pkl"), "rb") as f:
            self.user_top_genres = pickle.load(f)

        with open(self._path("user_seen.pkl"), "rb") as f:
            self.user_seen = pickle.load(f)

        with open(self._path("config.json")) as f:
            self.config = json.load(f)

        # Popularity fallback (Section 10 / 22) reconstructed from
        # movie_features.csv's bayesian_score column -- this table itself
        # was never exported separately, no need to duplicate it.
        self.popularity_ranking = (
            self.movie_features.sort_values("bayesian_score", ascending=False).index.to_numpy()
        )

    def global_mean_rating(self) -> float:
        return float(self.config["global_mean_rating"])

    def feature_cols(self):
        return self.config["feature_cols"]


@lru_cache(maxsize=1)
def get_store(artifact_dir: str = "artifacts") -> ArtifactStore:
    """Cached loader -- call this everywhere instead of constructing
    ArtifactStore directly, so the (fairly large) model files are only
    read from disk once per process."""
    return ArtifactStore(artifact_dir)
