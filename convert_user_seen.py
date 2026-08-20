import pickle
import numpy as np

print("Loading original user_seen.pkl (this will use ~1.7GB RAM briefly)...")
with open("artifacts/user_seen.pkl", "rb") as f:
    user_seen = pickle.load(f)

print(f"Converting {len(user_seen)} users to compact numpy format...")
compact = {
    int(user_id): np.array(sorted(movie_ids), dtype=np.int32)
    for user_id, movie_ids in user_seen.items()
}

with open("artifacts/user_seen_compact.pkl", "wb") as f:
    pickle.dump(compact, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Done. Wrote artifacts/user_seen_compact.pkl")