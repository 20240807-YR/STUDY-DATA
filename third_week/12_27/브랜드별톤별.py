# ============================================================
# Brand × Tone Cluster Centroid 생성
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CLUSTER_PATH = DATA_DIR / "brand_tone_cluster.csv"
EMBED_PATH = DATA_DIR / "brand_tone_embeddings.csv"

OUT_CSV = DATA_DIR / "brand_tone_centroid_profile.csv"
OUT_NPY = DATA_DIR / "brand_tone_centroid_vectors.npy"

cluster_df = pd.read_csv(CLUSTER_PATH)
embed_df = pd.read_csv(EMBED_PATH)

# 정규화
cluster_df["brand"] = cluster_df["brand"].astype(str).str.strip()
embed_df["tone_id"] = embed_df["tone_id"].astype(str).str.strip()

vec_cols = embed_df.columns.difference(["tone_id"])

rows = []
vectors = []

for (brand, cluster_id), g in cluster_df.groupby(["brand", "brand_tone_cluster"]):
    tone_names = g["brand"].tolist()

    sub = embed_df[embed_df["tone_id"].isin(tone_names)]
    if sub.empty:
        continue

    centroid = sub[vec_cols].values.mean(axis=0)
    vectors.append(centroid)

    rows.append({
        "brand": brand,
        "brand_tone_cluster": cluster_id,
        "tone_count": len(sub)
    })

X = np.array(vectors, dtype="float32")

np.save(OUT_NPY, X)
pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print("="*60)
print("brand tone centroid 생성 완료")
print("rows:", len(rows))
print("dim:", X.shape[1])
print("CSV:", OUT_CSV)
print("NPY:", OUT_NPY)
print("="*60)