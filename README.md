# MovieLens 25M — Recommendation Engine (source code)

Production-style two-stage movie recommender: ALS collaborative filtering for
candidate generation, a logistic regression ranker for scoring, with a
popularity fallback for cold-start users. Built and evaluated end-to-end in
a Kaggle/Colab notebook; this repo is the deployable source-code version of
that same pipeline.

## Project structure

```
movielens-recommender/
├── artifacts/              <- trained model files (see setup below)
├── src/recommender/
│   ├── artifacts.py        <- loads all model/data files once, cached
│   ├── features.py         <- per-(user, movie) feature computation
│   ├── candidates.py       <- ALS-based candidate generation
│   └── recommend.py        <- the public recommend_movies() function
├── scripts/
│   └── verify_artifacts.py <- run this first after setup
├── tests/
│   └── test_recommender.py
└── requirements.txt
```

## Setup

1. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Copy the **entire `artifacts/` folder you downloaded from your Kaggle
   notebook's Output tab** into this repo's `artifacts/` directory, so it
   contains: `als_model.pkl`, `ranking_model.pkl`, `id_mappings.pkl`,
   `movies.csv`, `user_features.csv`, `movie_features.csv`,
   `user_top_genres.pkl`, `user_seen.pkl`, `config.json`,
   `experiment_results.csv`.
3. Verify everything works:
   ```
   python scripts/verify_artifacts.py
   ```
   This loads the artifacts, recommends for a known user (should say
   `two_stage_model`) and an unknown user (should say
   `popularity_fallback`), and asserts both work correctly.
4. Run the test suite:
   ```
   pytest tests/
   ```

## Usage

```python
from recommender import get_store, recommend_movies

store = get_store("artifacts")
recs = recommend_movies(store, user_id=123, top_k=10)
print(recs)
```

## About committing the artifacts

The `artifacts/` folder is **not** gitignored on purpose: the Docker build
in the next phase needs these files present in the repo so they end up in
the built image. Check the total folder size after copying it in:

- **Under ~90MB total, no single file over ~90MB** → just `git add` and
  commit normally.
- **Larger than that** → GitHub will reject or warn on files over 100MB.
  Use [Git LFS](https://git-lfs.com/) for the `.pkl`/`.csv` files instead
  of committing them directly — `git lfs track "artifacts/*"` before your
  first commit.

## Known limitations (carried over from the notebook)

- Cold-start users (no training history) get popularity recommendations,
  not personalized ones — by design, see the notebook's Section 22.
- Brand-new movies with zero ratings won't be recommended until they
  accumulate some interactions.
- The ranking model is trained on a 300K-row sample of positive
  interactions, not the full dataset (see notebook Section 16) — a
  deliberate memory/runtime trade-off.

## Roadmap

```
[x] Phase 1 — This repo: clean source code wrapping the trained model
[ ] Phase 2 — FastAPI service exposing recommend_movies() over HTTP
[ ] Phase 3 — Docker containerization + local test
[ ] Phase 4 — Simple frontend/demo
[ ] Phase 5 — Deploy (Render/similar) + public URL
```
