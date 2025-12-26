# ============================================================
# Brand Analysis Text → Embedding
# ============================================================

import pandas as pd
import numpy as np
import requests
from pathlib import Path

BASE_DIR = Path("third_week/data_csv")
CSV_PATH = BASE_DIR / "total_brand_analysis.csv"

OUT_NPY = BASE_DIR / "brand_analysis_embeddings.npy"
OUT_CSV = BASE_DIR / "brand_analysis_embeddings.csv"

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"

# ------------------------------------------------------------
# 1) 로드
# ------------------------------------------------------------
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
print("[INFO] rows:", len(df))

# ------------------------------------------------------------
# 2) 임베딩용 텍스트 구성
# (컬럼명 네 CSV 기준 그대로 사용)
# ------------------------------------------------------------
def serialize(row):
    return f"""
브랜드: {row['브랜드']}
요약: {row['내용 요약']}
핵심 키워드: {row['핵심 키워드']}
AI 분석 톤: {row['AI 분석 톤']}
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

vectors = []
for i, t in enumerate(texts):
    print(f"[EMBED] {i+1}/{len(texts)}")
    vectors.append(embed(t))

X = np.array(vectors, dtype="float32")

# ------------------------------------------------------------
# 4) 저장
# ------------------------------------------------------------
np.save(OUT_NPY, X)

out_df = pd.DataFrame(X)
out_df.insert(0, "brand", df["브랜드"].values)
out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print("=" * 60)
print("brand_analysis_embeddings 생성 완료")
print("NPY:", OUT_NPY)
print("CSV:", OUT_CSV)
print("shape:", X.shape)
print("=" * 60)