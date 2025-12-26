# ==========================================
# Product Embedding 생성 스크립트 (Ollama)
# - 입력: data/amore_product_meta.csv
# - 출력: data/product_embeddings.npy
#         data/product_embeddings.csv
# - 모델: nomic-embed-text
# ==========================================

import requests
import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------
# 경로 설정
# ----------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

CSV_PATH = DATA_DIR / "amore_product_meta.csv"
NPY_OUT  = DATA_DIR / "product_embeddings.npy"
CSV_OUT  = DATA_DIR / "product_embeddings.csv"

OLLAMA_URL   = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "nomic-embed-text"

# ----------------------------
# 1) CSV 로드
# ----------------------------
if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV 파일 없음: {CSV_PATH}")

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
print("[INFO] Loaded product CSV:", df.shape)

# ----------------------------
# 2) 임베딩용 텍스트 직렬화
#    (상품 설명용으로 충분히 정보 밀도 확보)
# ----------------------------
def serialize_product(row: pd.Series) -> str:
    parts = []
    for col, val in row.items():
        if pd.isna(val):
            continue
        v = str(val).strip()
        if v == "":
            continue
        parts.append(f"{col}: {v}")
    return "\n".join(parts)

texts = df.apply(serialize_product, axis=1).tolist()

# ----------------------------
# 3) Ollama 임베딩 호출
# ----------------------------
def embed_text(text: str) -> list[float]:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": text
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["embedding"]

embeddings = []
for i, text in enumerate(texts):
    print(f"[EMBED] product {i+1}/{len(texts)}")
    vec = embed_text(text)
    embeddings.append(vec)

embeddings = np.array(embeddings, dtype="float32")

# ----------------------------
# 4) 저장
# ----------------------------
np.save(NPY_OUT, embeddings)

embed_df = pd.DataFrame(embeddings)
embed_df.insert(0, "product_index", range(len(embed_df)))
embed_df.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")

print("=" * 60)
print("제품 벡터화 완료")
print("NPY :", NPY_OUT)
print("CSV :", CSV_OUT)
print("shape:", embeddings.shape)
print("=" * 60)