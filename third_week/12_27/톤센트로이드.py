# ============================================================
# Brand Tone Cluster → Centroid Embedding (FIXED)
# ============================================================

import pandas as pd
import numpy as np
import requests
from pathlib import Path

# ----------------------------
# 경로
# ----------------------------
BASE_DIR = Path("third_week/12_27/data")
CSV_PATH = BASE_DIR / "brand_tone_cluster.csv"
OUT_NPY = BASE_DIR / "tone_centroid_embeddings.npy"
OUT_CSV = BASE_DIR / "tone_centroid_embeddings.csv"

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"

# ----------------------------
# 로드
# ----------------------------
df = pd.read_csv(CSV_PATH)

required_cols = {"brand", "brand_tone_cluster", "brand_position"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"CSV 컬럼 부족: {missing}")

print("[INFO] tone rows:", len(df))

# ----------------------------
# 직렬화
# ----------------------------
def serialize(row):
    return f"""
tone_name: {row['brand']}
tone_cluster_id: {row['brand_tone_cluster']}
brand_position: {row['brand_position']}
"""

def embed(text):
    r = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": text},
        timeout=60
    )
    r.raise_for_status()
    return r.json()["embedding"]

# ----------------------------
# 임베딩
# ----------------------------
vectors = []
for i, row in df.iterrows():
    print(f"[EMBED] {i+1}/{len(df)} | {row['brand']}")
    vectors.append(embed(serialize(row)))

X = np.array(vectors, dtype="float32")

# ----------------------------
# 저장
# ----------------------------
np.save(OUT_NPY, X)

out_df = pd.DataFrame(X)
out_df.insert(0, "brand", df["brand"].values)
out_df.insert(1, "brand_tone_cluster", df["brand_tone_cluster"].values)
out_df.insert(2, "brand_position", df["brand_position"].values)

out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print("=" * 60)
print("tone centroid embedding 생성 완료")
print("rows:", len(out_df))
print("dim:", X.shape[1])
print("NPY:", OUT_NPY)
print("CSV:", OUT_CSV)
print("=" * 60)