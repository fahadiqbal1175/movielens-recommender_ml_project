import os
import pickle
import psutil
import pandas as pd

process = psutil.Process(os.getpid())

def mem_mb():
    return process.memory_info().rss / (1024 * 1024)

def measure(label, load_fn):
    before = mem_mb()
    obj = load_fn()
    after = mem_mb()
    print(f"{label:25s}  disk n/a   ->  RAM delta: {after - before:8.1f} MB   (total RSS now: {after:.1f} MB)")
    return obj

ART = "artifacts"

print(f"Starting RSS: {mem_mb():.1f} MB\n")

als_model = measure("als_model.pkl", lambda: pickle.load(open(os.path.join(ART, "als_model.pkl"), "rb")))
ranking_model = measure("ranking_model.pkl", lambda: pickle.load(open(os.path.join(ART, "ranking_model.pkl"), "rb")))
id_mappings = measure("id_mappings.pkl", lambda: pickle.load(open(os.path.join(ART, "id_mappings.pkl"), "rb")))
movies = measure("movies.csv", lambda: pd.read_csv(os.path.join(ART, "movies.csv")))
user_features = measure("user_features.csv", lambda: pd.read_csv(os.path.join(ART, "user_features.csv"), index_col=0))
movie_features = measure("movie_features.csv", lambda: pd.read_csv(os.path.join(ART, "movie_features.csv"), index_col=0))
user_top_genres = measure("user_top_genres.pkl", lambda: pickle.load(open(os.path.join(ART, "user_top_genres.pkl"), "rb")))
user_seen = measure("user_seen.pkl", lambda: pickle.load(open(os.path.join(ART, "user_seen.pkl"), "rb")))

print(f"\nFinal RSS: {mem_mb():.1f} MB")

# Also show us the actual structure of user_seen so we know exactly what we're dealing with
sample_key = next(iter(user_seen))
sample_val = user_seen[sample_key]
print(f"\nuser_seen structure: dict with {len(user_seen)} keys")
print(f"Sample key: {sample_key} (type: {type(sample_key)})")
print(f"Sample value type: {type(sample_val)}, length: {len(sample_val) if hasattr(sample_val, '__len__') else 'n/a'}")
print(f"Sample value preview: {list(sample_val)[:5] if hasattr(sample_val, '__iter__') else sample_val}")