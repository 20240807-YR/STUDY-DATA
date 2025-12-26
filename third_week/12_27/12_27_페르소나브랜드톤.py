# ============================================================
# Persona × Brand Tone Strategy (FINAL FIX)
# ============================================================

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# PATH
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PERSONA_VEC_PATH = DATA_DIR / "persona_vectors.npy"
SIM_PATH         = DATA_DIR / "persona_product_similarity_full.csv"
TONE_VEC_PATH    = DATA_DIR / "tone_vectors.pkl"

OUT_PATH = DATA_DIR / "persona_brand_tone_strategy.csv"

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------
persona_vecs = np.load(PERSONA_VEC_PATH)
sim_df = pd.read_csv(SIM_PATH)

with open(TONE_VEC_PATH, "rb") as f:
    tone_vectors = pickle.load(f)

print("[INFO] persona_vectors:", persona_vecs.shape)
print("[INFO] sim_df columns:", sim_df.columns.tolist())
print("[INFO] tone brands:", list(tone_vectors.keys()))

# ------------------------------------------------------------
# brand 컬럼 자동 인식
# ------------------------------------------------------------
if "brand" not in sim_df.columns:
    raise RuntimeError(f"[ERROR] brand 컬럼 없음: {sim_df.columns.tolist()}")

# ------------------------------------------------------------
# persona × brand aggregation
# ------------------------------------------------------------
agg_df = (
    sim_df
    .dropna(subset=["brand"])
    .groupby(["persona_id", "brand"])
    .agg(
        avg_similarity=("similarity", "mean"),
        product_cnt=("similarity", "count")
    )
    .reset_index()
)

print("[INFO] agg_df:", agg_df.shape)

# ------------------------------------------------------------
# Persona × Brand Tone Cluster 매칭
# tone_vectors = {cluster_name: centroid_vector}
# ------------------------------------------------------------
rows = []

for _, r in agg_df.iterrows():
    persona_id = r["persona_id"]
    brand = str(r["brand"]).strip()

    if brand not in tone_vectors:
        continue

    persona_idx = int(persona_id.split("_")[1]) - 1
    persona_vec = persona_vecs[persona_idx].reshape(1, -1)

    cluster_names = list(tone_vectors.keys())
    cluster_vecs  = np.vstack([tone_vectors[c] for c in cluster_names])

    sims = cosine_similarity(persona_vec, cluster_vecs)[0]
    best_cluster = cluster_names[int(np.argmax(sims))]

    rows.append({
        "persona_id": persona_id,
        "brand": brand,
        "brand_tone_cluster": best_cluster,
        "avg_similarity": float(r["avg_similarity"]),
        "product_cnt": int(r["product_cnt"])
    })

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------
out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("=" * 60)
print("✅ persona_brand_tone_strategy.csv 생성 완료")
print("ROWS:", len(out_df))
print("PATH:", OUT_PATH)
print("=" * 60)