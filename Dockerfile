# MovieLens Recommender API — container image
#
# Base: slim Python image (not alpine — implicit/scipy/scikit-learn ship
# manylinux wheels built against glibc, not musl, so alpine would force a
# from-source build of the whole scientific stack).
FROM python:3.11-slim

# libgomp1 is the OpenMP runtime that `implicit`'s compiled ALS routines
# need at import/run time. It's not pulled in automatically by pip because
# the wheel only *links* against it -- it doesn't vendor it. Without this,
# `import implicit` (triggered indirectly via recommender/artifacts.py)
# fails inside the container even though it works fine locally, if your
# local machine happens to already have libgomp installed system-wide.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first, separately from app code, so `docker build`
# reuses this layer on every rebuild that only touches app/src/artifacts --
# which is the common case once requirements.txt stabilizes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code and trained model artifacts. Order matters for cache
# invalidation: src/ and app/ change more often than artifacts/, but
# artifacts/ is by far the largest layer (~116MB), so it's placed last --
# editing main.py shouldn't force Docker to re-lay-down the model files.
COPY src/ ./src/
COPY app/ ./app/
COPY artifacts/ ./artifacts/

# Not strictly enforced by Docker itself, but documents the port the app
# listens on for anyone reading the Dockerfile, and is picked up by
# `docker run -P` / some orchestrators.
EXPOSE 8000

# Basic container-level liveness check, independent of any orchestrator
# config. Hits the same /health endpoint app/main.py already exposes.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

# --host 0.0.0.0 is required -- uvicorn's default (127.0.0.1) only accepts
# connections from inside the container's own network namespace, which
# would make the API unreachable from the host even with -p 8000:8000.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
