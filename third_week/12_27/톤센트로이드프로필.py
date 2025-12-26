# ============================================================
# Tone Cluster → Centroid Profile 생성 (FIXED)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

TONE_CLUSTER_PATH = DATA_DIR / "brand_tone_cluster.csv"
TONE_EMBED_CSV = DATA_DIR / "brand_tone_embeddings.csv"
OUT_CSV = DATA_DIR / "tone_centroid_profile.csv"
OUT_NPY = DATA_DIR / "tone_centroid_profile.npy"

# ------------------------------------------------------------
# 로드
# ------------------------------------------------------------
cluster_df = pd.read_csv(TONE_CLUSTER_PATH)
embed_df = pd.read_csv(TONE_EMBED_CSV)

# 정규화
cluster_df["brand"] = cluster_df["brand"].astype(str).str.strip()
embed_df["tone_id"] = embed_df["tone_id"].astype(str).str.strip()

# embedding 컬럼만
vec_cols = embed_df.columns.difference(["tone_id"])

# ------------------------------------------------------------
# cluster 기준 centroid 계산
# ------------------------------------------------------------
rows = []
centroids = []

for cluster_id, g in cluster_df.groupby("brand_tone_cluster"):
    tone_ids = g["brand"].unique().tolist()  # ← 여기서 brand = tone_id 의미

    sub = embed_df[embed_df["tone_id"].isin(tone_ids)]
    if sub.empty:
        continue

    centroid = sub[vec_cols].values.mean(axis=0)
    centroids.append(centroid)

    rows.append({
        "brand_tone_cluster": cluster_id,
        "tone_count": len(sub),
        "tones": ", ".join(tone_ids)
    })

# ------------------------------------------------------------
# 저장
# ------------------------------------------------------------
centroids = np.array(centroids, dtype="float32")
np.save(OUT_NPY, centroids)

pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print("=" * 60)
print("tone_centroid_profile 생성 완료")
print("clusters:", centroids.shape[0])
print("dim:", centroids.shape[1])
print("CSV:", OUT_CSV)
print("NPY:", OUT_NPY)
print("=" * 60)