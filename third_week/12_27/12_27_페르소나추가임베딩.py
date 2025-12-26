# ==========================================
# Persona Embedding 생성 스크립트 (Ollama)
# - 입력: data/persona_meta_v2.csv
# - 출력: data/persona_embeddings.npy
#         data/persona_embeddings.csv
# - 임베딩 모델: Ollama (nomic-embed-text)
# ==========================================

import os
import json
import requests
import numpy as np
import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent   # third_week/12_27
DATA_DIR = BASE_DIR / "data"

CSV_PATH = DATA_DIR / "persona_meta_v2.csv"
NPY_OUT  = DATA_DIR / "persona_embeddings.npy"
CSV_OUT  = DATA_DIR / "persona_embeddings.csv"

OLLAMA_URL   = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "nomic-embed-text"

# ----------------------------
# 1) CSV 로드
# ----------------------------
if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV 파일 없음: {CSV_PATH}")

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
print("[INFO] Loaded CSV:", df.shape)

# ----------------------------
# 2) 임베딩용 텍스트 직렬화
#    - persona_id 제외
#    - 컬럼:값 형태로 모두 연결
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
for idx, text in enumerate(texts):
    print(f"[EMBED] persona {idx+1}/{len(texts)}")
    vec = embed_text(text)
    embeddings.append(vec)

embeddings = np.array(embeddings, dtype="float32")

# ----------------------------
# 4) 결과 저장
# ----------------------------
np.save(NPY_OUT, embeddings)

embed_df = pd.DataFrame(embeddings)
embed_df.insert(0, "persona_id", df["persona_id"].values)
embed_df.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")

print("=" * 50)
print("임베딩 완료")
print("NPY:", NPY_OUT)
print("CSV:", CSV_OUT)
print("shape:", embeddings.shape)
print("=" * 50)