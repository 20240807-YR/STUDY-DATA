# ============================================================
# Persona × Brand Tone Strategy (FINAL FIXED)
# ============================================================

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PERSONA_VEC_PATH = DATA_DIR / "persona_vectors.npy"
SIM_PATH = DATA_DIR / "persona_product_similarity_full.csv"
TONE_VEC_PATH = DATA_DIR / "tone_vectors.pkl"

OUT_PATH = DATA_DIR / "persona_brand_tone_strategy.csv"

# ------------------------------------------------------------
# load
# ------------------------------------------------------------
persona_vecs = np.load(PERSONA_VEC_PATH)
sim_df = pd.read_csv(SIM_PATH)

with open(TONE_VEC_PATH, "rb") as f:
    tone_vectors = pickle.load(f)

# ------------------------------------------------------------
# sanity check
# ------------------------------------------------------------
required_cols = {"persona_id", "brand", "similarity"}
missing = required_cols - set(sim_df.columns)
if missing:
    raise RuntimeError(f"❌ SIM CSV 컬럼 부족: {missing}")

# brand 정규화
sim_df["brand"] = sim_df["brand"].astype(str).str.strip()

# ------------------------------------------------------------
# 집계
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

# ------------------------------------------------------------
# 브랜드 톤 클러스터 매칭
# ------------------------------------------------------------
rows = []

for _, r in agg_df.iterrows():
    persona_id = r["persona_id"]
    brand = r["brand"]

    if brand not in tone_vectors:
        continue

    persona_idx = int(persona_id.split("_")[1]) - 1
    persona_vec = persona_vecs[persona_idx].reshape(1, -1)

    clusters = tone_vectors[brand]  # {cluster_id: vector}
    cluster_ids = list(clusters.keys())
    cluster_vecs = np.vstack([clusters[c] for c in cluster_ids])

    sims = cosine_similarity(persona_vec, cluster_vecs)[0]
    best_cluster = cluster_ids[int(np.argmax(sims))]

    rows.append({
        "persona_id": persona_id,
        "brand": brand,
        "brand_tone_cluster": best_cluster,
        "avg_similarity": float(r["avg_similarity"]),
        "product_cnt": int(r["product_cnt"])
    })

# ------------------------------------------------------------
# output
# ------------------------------------------------------------
out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("=" * 50)
print("✅ persona_brand_tone_strategy.csv 생성 완료")
print("ROWS:", len(out_df))
print("PATH:", OUT_PATH)
print("=" * 50)