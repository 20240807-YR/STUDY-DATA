# ============================================================
# Brand Tone Clustering (KMeans)
# - input : data/tone_vectors.pkl   (brand -> vector)
# - output: data/brand_tone_cluster.csv
# ============================================================

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans

# ----------------------------
# 설정
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

TONE_PKL_PATH = DATA_DIR / "tone_vectors.pkl"
OUT_CSV_PATH  = DATA_DIR / "brand_tone_cluster.csv"

N_CLUSTERS = 4          # ← 필요하면 3~6으로 조정 가능
RANDOM_STATE = 42

# ----------------------------
# 1. tone_vectors.pkl 로드
# ----------------------------
if not TONE_PKL_PATH.exists():
    raise FileNotFoundError(f"tone_vectors.pkl 없음: {TONE_PKL_PATH}")

with open(TONE_PKL_PATH, "rb") as f:
    tone_dict = pickle.load(f)

# 기대 구조: { brand_name: np.array([...]) }
assert isinstance(tone_dict, dict), "tone_vectors.pkl 구조가 dict 아님"

brands = list(tone_dict.keys())
vectors = np.vstack([tone_dict[b] for b in brands])

print("[INFO] brand count:", len(brands))
print("[INFO] vector shape:", vectors.shape)

# ----------------------------
# 2. KMeans 클러스터링
# ----------------------------
kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=10
)

cluster_labels = kmeans.fit_predict(vectors)

# ----------------------------
# 3. 결과 CSV 생성
# ----------------------------
df_out = pd.DataFrame({
    "brand": brands,
    "brand_tone_cluster": cluster_labels
})

df_out.to_csv(OUT_CSV_PATH, index=False, encoding="utf-8-sig")

print("=" * 50)
print("브랜드 톤 클러스터링 완료")
print("OUTPUT:", OUT_CSV_PATH)
print(df_out["brand_tone_cluster"].value_counts().sort_index())
print("=" * 50)