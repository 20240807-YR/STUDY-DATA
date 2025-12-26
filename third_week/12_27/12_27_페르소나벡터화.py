# ============================================================
# Persona Vectorization Script (Ollama)
# - input : data/persona_meta_v2.csv
# - output: data/persona_vectors.npy
#           data/persona_vectors.csv
# - model : nomic-embed-text
# ============================================================

import requests
import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------
# 경로 설정 (파일 위치 기준)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CSV_PATH = DATA_DIR / "persona_meta_v2.csv"
NPY_OUT  = DATA_DIR / "persona_vectors.npy"
CSV_OUT  = DATA_DIR / "persona_vectors.csv"

OLLAMA_URL   = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "nomic-embed-text"

# ----------------------------
# CSV 로드
# ----------------------------
if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV 없음: {CSV_PATH}")

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
print("[INFO] persona csv:", df.shape)

# ----------------------------
# 페르소나 직렬화
# - persona_id, persona_name 제외
# - 임베딩 입력은 '의미 텍스트'만
# ----------------------------
def serialize_persona(row: pd.Series) -> str:
    parts = []
    for col, val in row.items():
        if col in ["persona_id", "persona_name"]:
            continue
        if pd.isna(val):
            continue
        v = str(val).strip()
        if v == "" or v == "-":
            continue
        parts.append(f"{col}: {v}")
    return "\n".join(parts)

texts = df.apply(serialize_persona, axis=1).tolist()

# ----------------------------
# Ollama 임베딩 호출
# ----------------------------
def embed_text(text: str) -> list:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": text
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["embedding"]

vectors = []
for i, text in enumerate(texts):
    print(f"[EMBED] persona {i+1}/{len(texts)}")
    vec = embed_text(text)
    vectors.append(vec)

vectors = np.array(vectors, dtype="float32")

# ----------------------------
# 저장
# ----------------------------
np.save(NPY_OUT, vectors)

vec_df = pd.DataFrame(vectors)
vec_df.insert(0, "persona_id", df["persona_id"].values)
vec_df.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")

print("=" * 60)
print("페르소나 벡터화 완료")
print("NPY:", NPY_OUT)
print("CSV:", CSV_OUT)
print("shape:", vectors.shape)
print("=" * 60)