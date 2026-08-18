"""
FastAPI service wrapping recommend_movies() as an HTTP endpoint.

Artifacts (the ~116MB of model files under artifacts/) are loaded exactly
once, at process startup, via the lifespan handler below -- not on every
request. That's the same principle recommender.get_store()'s lru_cache
enforces for direct script usage (see scripts/verify_artifacts.py); here
it matters even more, since a per-request load would make every API call
take as long as the whole verify script does.

Run locally:
    uvicorn app.main:app --reload

Interactive docs (once running):
    http://127.0.0.1:8000/docs
"""

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Path as PathParam, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from recommender import ArtifactStore, get_store, recommend_movies  # noqa: E402

from app.schemas import (  # noqa: E402
    ErrorResponse,
    HealthResponse,
    MovieRecommendation,
    RecommendResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recommender_api")

ARTIFACT_DIR = str(REPO_ROOT / "artifacts")
DEFAULT_TOP_K = 10
MAX_TOP_K = 100

# Plain dict rather than a module-level global we reassign -- reassigning a
# module-level name from inside an async function needs `global`, which is
# easy to get wrong; mutating a dict in place avoids that entirely.
_state: Dict[str, ArtifactStore] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading artifacts from %s ...", ARTIFACT_DIR)
    start = time.perf_counter()
    try:
        _state["store"] = get_store(ARTIFACT_DIR)
    except Exception:
        logger.exception("Failed to load artifacts from %s", ARTIFACT_DIR)
        raise
    logger.info(
        "Artifacts loaded in %.2fs (%d users, %d movies)",
        time.perf_counter() - start,
        len(_state["store"].user_id_to_idx),
        len(_state["store"].movies),
    )
    yield
    _state.clear()


app = FastAPI(
    title="MovieLens Recommender API",
    description=(
        "Two-stage movie recommendation service (ALS candidate generation "
        "-> logistic regression ranker), with a popularity fallback for "
        "cold-start / unknown users."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# The demo frontend is served by this same app (see the static mount at the
# bottom of this file), so cross-origin requests aren't needed in production.
# CORS is left open here anyway so the frontend can also be pointed at this
# API from a different origin during local development (e.g. a separate
# `python -m http.server` for the static files while iterating on app.js).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _get_store() -> ArtifactStore:
    store = _state.get("store")
    if store is None:
        # Only reachable if a request somehow lands before lifespan startup
        # finishes, or after shutdown has cleared the state.
        raise HTTPException(status_code=503, detail="Model artifacts are still loading. Try again shortly.")
    return store


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness/readiness probe for Docker/Render. Returns 200 with
    artifacts_loaded=False while startup is still in progress, rather than
    failing outright, so orchestrators can distinguish 'starting up' from
    'crashed'."""
    store = _state.get("store")
    return HealthResponse(
        status="ok" if store is not None else "loading",
        artifacts_loaded=store is not None,
        n_users=len(store.user_id_to_idx) if store is not None else 0,
        n_movies=len(store.movies) if store is not None else 0,
    )


@app.get(
    "/recommend/{user_id}",
    response_model=RecommendResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Invalid user_id or top_k."},
        503: {"model": ErrorResponse, "description": "Artifacts still loading."},
    },
    tags=["recommendations"],
)
def recommend(
    user_id: int = PathParam(..., ge=1, description="MovieLens userId. Unknown IDs fall back to popularity."),
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K, description="Number of recommendations to return."),
) -> RecommendResponse:
    """Recommend top_k movies for user_id. Known users get two-stage
    (ALS + ranker) recommendations; unknown/cold-start users transparently
    fall back to a popularity ranking -- see recommend_movies() for the
    actual logic, this endpoint is a thin wrapper around it."""
    store = _get_store()

    try:
        recs = recommend_movies(store, user_id=user_id, top_k=top_k)
    except Exception:
        logger.exception("recommend_movies failed for user_id=%s top_k=%s", user_id, top_k)
        raise HTTPException(status_code=500, detail="Failed to generate recommendations.") from None

    items = [
        MovieRecommendation(
            movie_id=int(row.movieId),
            title=row.title,
            genres=row.genres,
            score=None if row.score is None else float(row.score),
            source=row.source,
        )
        for row in recs.itertuples(index=False)
    ]

    source: Optional[str] = items[0].source if items else "popularity_fallback"

    return RecommendResponse(
        user_id=user_id,
        top_k=top_k,
        count=len(items),
        source=source,
        recommendations=items,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all so an unexpected error (e.g. a corrupt artifact file
    surfacing mid-request) returns a clean 500 with no stack trace leaked
    to the client, instead of an unhandled-error crash response."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


# Mounted last and at "/" so it never shadows the API routes above --
# Starlette matches routes in the order they were added, so /health and
# /recommend/{user_id} are always resolved first. html=True serves
# static/index.html for "/" and for any unmatched path, which is what we
# want for a single-page demo.
STATIC_DIR = REPO_ROOT / "app" / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="frontend")
