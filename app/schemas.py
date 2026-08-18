"""
Pydantic models for the FastAPI service's request/response bodies. Kept
separate from main.py so the API's public shape is easy to scan (and easy
to reuse from tests) without wading through routing/startup logic.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class MovieRecommendation(BaseModel):
    movie_id: int = Field(description="MovieLens movieId.")
    title: str
    genres: str = Field(description="Pipe-separated genre list, as stored in movies.csv (e.g. 'Comedy|Romance').")
    score: Optional[float] = Field(
        default=None,
        description=(
            "Ranking model's predicted probability of a positive interaction. "
            "None for popularity-fallback recommendations, which aren't scored by the ranker."
        ),
    )
    source: str = Field(description="'two_stage_model' or 'popularity_fallback'.")


class RecommendResponse(BaseModel):
    user_id: int
    top_k: int
    count: int = Field(description="Number of recommendations actually returned (may be less than top_k).")
    source: str = Field(description="'two_stage_model' for a known user, 'popularity_fallback' for cold-start.")
    recommendations: List[MovieRecommendation]


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' once artifacts are loaded, 'loading' during startup.")
    artifacts_loaded: bool
    n_users: int
    n_movies: int


class ErrorResponse(BaseModel):
    detail: str
