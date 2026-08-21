# MovieLens 25M — Production-Style Movie Recommendation System

A two-stage movie recommender trained on the full **MovieLens 25M** dataset (25M ratings, 62K movies, 137K users), served behind a FastAPI backend with a cinema/box-office themed frontend, deployed live on Render.

**🎬 Live demo:** https://movielens-recommender-a4je.onrender.com

**Source:** https://github.com/fahadiqbal1175/movielens-recommender_ml_project

![Demo screenshot](docs/screenshot.gif)

---

## What this is

Most "movie recommender" portfolio projects stop at a similarity matrix in a notebook. This one goes further: the model is trained end-to-end on the full 25M-rating dataset, wrapped in a proper Python package, exposed over a REST API, containerized, and deployed to a live public URL — including hitting and fixing a real production issue (an out-of-memory crash) along the way.

## Architecture

```
Request → FastAPI → recommend_movies()
                          │
              ┌───────────┴───────────┐
              │                       │
        known user?              unknown / cold-start user?
              │                       │
   Stage 1: ALS candidate      Popularity fallback
   generation (collaborative         │
   filtering)                        │
              │                      │
   Stage 2: logistic regression      │
   ranker scores each candidate      │
              │                      │
              └───────────┬──────────┘
                     top-K results
```

- **Stage 1 — Candidate generation:** ALS (Alternating Least Squares) matrix factorization over the user–item interaction matrix produces a shortlist of plausible movies per user.
- **Stage 2 — Ranking:** a logistic regression model scores each candidate using user/movie/interaction features, and the top-K by score are returned.
- **Cold-start handling:** users with no training history (or fake/unknown IDs) transparently fall back to a popularity-based ranking instead of failing — `source: "popularity_fallback"` vs `source: "two_stage_model"` in the response.

Full training pipeline (EDA, temporal train/test split, baselines, ALS, ranking model, evaluation with Precision@K / Recall@K / NDCG@K / Hit Rate@K, cold-start analysis) was built and evaluated in a Kaggle/Colab notebook; this repo is the deployable source-code version of that pipeline.

## Project structure

```
movielens-recommender/
├── artifacts/              trained model files (als_model.pkl, ranking_model.pkl,
│                            id_mappings.pkl, movies.csv, user_features.csv,
│                            movie_features.csv, user_top_genres.pkl, user_seen.pkl, ...)
├── src/recommender/
│   ├── artifacts.py         loads all model/data files once, cached
│   ├── features.py          per-(user, movie) feature computation
│   ├── candidates.py        ALS-based candidate generation
│   └── recommend.py         the public recommend_movies() function
├── app/
│   ├── main.py               FastAPI app (loads artifacts once at startup)
│   ├── schemas.py             request/response models
│   └── static/                cinema-themed frontend (served by the same app)
├── scripts/
│   └── verify_artifacts.py  sanity-checks a known + unknown user end-to-end
├── tests/
│   ├── test_recommender.py
│   └── test_api.py
├── Dockerfile
└── requirements.txt
```

## API

| Endpoint | Description |
|---|---|
| `GET /health` | Service + artifact status |
| `GET /recommend/{user_id}?top_k=10` | Top-K recommendations for a user |
| `GET /` | Frontend demo |
| `GET /docs` | Interactive Swagger docs |

```bash
curl "https://movielens-recommender-a4je.onrender.com/health"
# {"status":"ok","artifacts_loaded":true,"n_users":137840,"n_movies":62423}

curl "https://movielens-recommender-a4je.onrender.com/recommend/123?top_k=5"
# {"user_id":123,"top_k":5,"count":5,"source":"two_stage_model","recommendations":[{"movie_id":2353,"title":"Enemy of the State (1998)","genres":"Action|Thriller","score":0.9995160482680447,"source":"two_stage_model"},{"movie_id":1573,"title":"Face/Off (1997)","genres":"Action|Crime|Drama|Thriller","score":0.9993302149093608,"source":"two_stage_model"},{"movie_id":357,"title":"Four Weddings and a Funeral (1994)","genres":"Comedy|Romance","score":0.9980549945796425,"source":"two_stage_model"},{"movie_id":1370,"title":"Die Hard 2 (1990)","genres":"Action|Adventure|Thriller","score":0.9980038848159132,"source":"two_stage_model"},{"movie_id":368,"title":"Maverick (1994)","genres":"Adventure|Comedy|Western","score":0.9974780779728559,"source":"two_stage_model"}]}
# {"user_id":999999999,"top_k":5,"count":5,"source":"popularity_fallback","recommendations":[{"movie_id":318,"title":"Shawshank Redemption, The (1994)","genres":"Crime|Drama","score":null,"source":"popularity_fallback"},{"movie_id":858,"title":"Godfather, The (1972)","genres":"Crime|Drama","score":null,"source":"popularity_fallback"},{"movie_id":50,"title":"Usual Suspects, The (1995)","genres":"Crime|Mystery|Thriller","score":null,"source":"popularity_fallback"},{"movie_id":527,"title":"Schindler's List (1993)","genres":"Drama|War","score":null,"source":"popularity_fallback"},{"movie_id":1221,"title":"Godfather: Part II, The (1974)","genres":"Crime|Drama","score":null,"source":"popularity_fallback"}]}
```

Invalid input (`user_id <= 0`, non-integer `user_id`, `top_k` outside 1–100) returns `422` with details on what was wrong.

## Running locally

**Python:**
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# copy the artifacts/ folder (from the training notebook's output) into place, then:
python scripts/verify_artifacts.py   # sanity check
uvicorn app.main:app --reload
```
Visit `http://127.0.0.1:8000` for the frontend or `http://127.0.0.1:8000/docs` for the API docs.

**Docker:**
```bash
docker build -t movielens-recommender .
docker run -p 8000:8000 movielens-recommender
```

## Engineering notes: the memory-optimization story

At first the app was using ~1.98GB RAM. A memory-profiling script traced to a single file, `user_seen.pkl`: a pandas Series where each value was a Python `set`, and Python `set` objects carry very high per-object memory overhead at this scale (137K users × per-movie sets).

**Fix:** `user_seen` was converted from a `Series` of `set` objects into a `dict` of compact NumPy `int32` arrays, with `candidates.py` and `recommend.py` updated to work against arrays/`None` instead of assuming Python sets. Combined with pinning `scikit-learn==1.6.1` to match the version the model was actually trained with, this brought the resident memory footprint down to **~447.7MB** — comfortably under the 512MB limit — while keeping identical recommendation output (verified by the full `pytest` suite, 11/11 passing).

## Known limitations

- Cold-start users (no training history) get popularity recommendations, not personalized ones — by design.
- Brand-new movies with zero ratings won't be recommended until they accumulate some interactions.
- The ranking model is trained on a 300K-row sample of positive interactions rather than the full dataset — a deliberate memory/runtime trade-off.
- Free-tier hosting means the service spins down when idle; the first request afterward takes 30–60s to cold-start.

## Roadmap

```
[x] Phase 1 — Source code wrapping the trained model
[x] Phase 2 — FastAPI service exposing recommend_movies() over HTTP
[x] Phase 3 — Docker containerization
[x] Phase 4 — Cinema-themed frontend demo
[x] Phase 5 — Deployed live on Render, memory-optimized, public URL
```
