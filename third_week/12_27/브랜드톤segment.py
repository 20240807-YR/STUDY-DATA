# ============================================================
# Brand Tone Segmented CSV 생성 (12_27/data 고정)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

# ----------------------------
# 경로 (현재 파일 = third_week/12_27/*.py 에서 실행 가정)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent          # third_week/12_27
DATA_DIR = BASE_DIR / "data"                       # third_week/12_27/data

BRAND_ANALYSIS_PATH = DATA_DIR / "total_brand_analysis.csv"
TONE_CLUSTER_PATH   = DATA_DIR / "brand_tone_cluster.csv"
EMBED_PATH          = DATA_DIR / "brand_analysis_embeddings.npy"

OUT_PATH            = DATA_DIR / "brand_tone_segmented.csv"

# ----------------------------
# 로드
# ----------------------------
if not BRAND_ANALYSIS_PATH.exists():
    raise FileNotFoundError(f"파일 없음: {BRAND_ANALYSIS_PATH}")
if not TONE_CLUSTER_PATH.exists():
    raise FileNotFoundError(f"파일 없음: {TONE_CLUSTER_PATH}")
if not EMBED_PATH.exists():
    raise FileNotFoundError(f"파일 없음: {EMBED_PATH}")

analysis_df = pd.read_csv(BRAND_ANALYSIS_PATH, encoding="utf-8-sig")
cluster_df  = pd.read_csv(TONE_CLUSTER_PATH, encoding="utf-8-sig")
embeddings  = np.load(EMBED_PATH)

print("[INFO] analysis rows:", len(analysis_df))
print("[INFO] embeddings shape:", embeddings.shape)
print("[INFO] cluster cols:", list(cluster_df.columns))
print("[INFO] analysis cols:", list(analysis_df.columns))

# ----------------------------
# 필수 컬럼 체크
# ----------------------------
required_cols = {"brand", "part", "content"}
missing = required_cols - set(analysis_df.columns)
if missing:
    raise ValueError(f"analysis CSV 컬럼 부족: {missing}")

if "brand" not in cluster_df.columns or "brand_tone_cluster" not in cluster_df.columns:
    raise ValueError("brand_tone_cluster.csv 에 'brand', 'brand_tone_cluster' 컬럼이 필요합니다.")

# ----------------------------
# 정규화
# ----------------------------
analysis_df["brand"] = analysis_df["brand"].astype(str).str.strip()
cluster_df["brand"]  = cluster_df["brand"].astype(str).str.strip()

# ----------------------------
# brand → tone_cluster 매핑
# ----------------------------
brand_to_cluster = cluster_df.set_index("brand")["brand_tone_cluster"].to_dict()

analysis_df["brand_tone_cluster"] = analysis_df["brand"].map(brand_to_cluster)
analysis_df = analysis_df.dropna(subset=["brand_tone_cluster"]).copy()
analysis_df["brand_tone_cluster"] = analysis_df["brand_tone_cluster"].astype(int)

# ----------------------------
# 결과 CSV 구성
# ----------------------------
out_df = analysis_df[["brand", "part", "content", "brand_tone_cluster"]].copy()
out_df = out_df.rename(columns={"content": "content_summary"})

# ----------------------------
# 저장
# ----------------------------
out_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("=" * 60)
print("brand_tone_segmented.csv 생성 완료")
print("ROWS:", len(out_df))
print("PATH:", OUT_PATH)
print("=" * 60)