# ============================================================
# Brand Tone Embedding 생성 (Ollama)
# - input :
#   data/brand_tone_definitions.csv
# - output:
#   data/brand_tone_embeddings.npy
#   data/brand_tone_embeddings.csv
# ============================================================

import requests
import numpy as np
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# 설정
# ------------------------------------------------------------
BASE_DIR = Path("third_week/12_27")
DATA_DIR = BASE_DIR / "data"

CSV_PATH = DATA_DIR / "brand_tone_definitions.csv"
NPY_OUT  = DATA_DIR / "brand_tone_embeddings.npy"
CSV_OUT  = DATA_DIR / "brand_tone_embeddings.csv"

OLLAMA_URL   = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "nomic-embed-text"

# ------------------------------------------------------------
# 1) CSV 로드
# ------------------------------------------------------------
if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV 파일 없음: {CSV_PATH}")

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
print("[INFO] Loaded tone definitions:", df.shape)

# ------------------------------------------------------------
# 2) 임베딩용 텍스트 구성
#    - tone_id + 설명 결합
# ------------------------------------------------------------
def serialize_tone(row: pd.Series) -> str:
    return (
        f"tone_id: {row['tone_id']}\n"
        f"preview: {row['description_preview']}\n"
        f"description: {row['full_description']}"
    )

texts = df.apply(serialize_tone, axis=1).tolist()

# ------------------------------------------------------------
# 3) Ollama 임베딩 호출
# ------------------------------------------------------------
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
    print(f"[EMBED] tone {i+1}/{len(texts)}")
    embeddings.append(embed_text(text))

embeddings = np.array(embeddings, dtype="float32")

# ------------------------------------------------------------
# 4) 저장
# ------------------------------------------------------------
np.save(NPY_OUT, embeddings)

embed_df = pd.DataFrame(embeddings)
embed_df.insert(0, "tone_id", df["tone_id"].values)
embed_df.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")

print("=" * 60)
print("Brand tone embedding 완료")
print("NPY :", NPY_OUT)
print("CSV :", CSV_OUT)
print("shape:", embeddings.shape)
print("=" * 60)