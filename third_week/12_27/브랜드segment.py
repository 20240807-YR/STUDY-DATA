# ============================================================
# Brand Tone Clustering Pipeline (FINAL)
# ============================================================

import numpy as np
import pandas as pd
import requests
from pathlib import Path
from sklearn.cluster import KMeans

# ------------------------------------------------------------
# 경로
# ------------------------------------------------------------
BASE_DIR = Path("third_week/12_27")
DATA_DIR = BASE_DIR / "data"

TONE_CSV = DATA_DIR / "brand_tone_definitions.csv"
OUT_CSV  = DATA_DIR / "brand_tone_cluster.csv"

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"

N_CLUSTERS = 4   # ← 지금은 4, 나중에 6~8로 늘려도 됨

# ------------------------------------------------------------
# 1) 로드
# ------------------------------------------------------------
df = pd.read_csv(TONE_CSV, encoding="utf-8-sig")
print("[INFO] tone rows:", len(df))

# ------------------------------------------------------------
# 2) 임베딩 텍스트
# ------------------------------------------------------------
def serialize(row):
    return f"""
tone_id: {row['tone_id']}
preview: {row['description_preview']}
description: {row['full_description']}
"""

texts = df.apply(serialize, axis=1).tolist()

# ------------------------------------------------------------
# 3) Ollama 임베딩
# ------------------------------------------------------------
def embed(text):
    r = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": text},
        timeout=60
    )
    r.raise_for_status()
    return r.json()["embedding"]

embeddings = []
for i, t in enumerate(texts):
    print(f"[EMBED] {i+1}/{len(texts)}")
    embeddings.append(embed(t))

X = np.array(embeddings, dtype="float32")

# ------------------------------------------------------------
# 4) KMeans
# ------------------------------------------------------------
kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    n_init="auto"
)
labels = kmeans.fit_predict(X)

df["brand_tone_cluster"] = labels

# ------------------------------------------------------------
# 5) 사람이 읽을 포지션 라벨 (고정)
# ------------------------------------------------------------
CLUSTER_LABEL_MAP = {
    0: "고기능/클리니컬 톤",
    1: "수분 중심 톤",
    2: "프리미엄/헤리티지 톤",
    3: "트렌디/컬러 중심 톤"
}

df["brand_position"] = df["brand_tone_cluster"].map(CLUSTER_LABEL_MAP)

# ------------------------------------------------------------
# 6) 최종 CSV
# ------------------------------------------------------------
out_df = df[["tone_id", "brand_tone_cluster", "brand_position"]] \
            .rename(columns={"tone_id": "brand"})

out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print("=" * 60)
print("brand_tone_cluster.csv 생성 완료")
print("PATH:", OUT_CSV)
print(out_df.head())
print("=" * 60)