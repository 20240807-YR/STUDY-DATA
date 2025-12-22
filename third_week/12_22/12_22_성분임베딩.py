# ingredient_embedding_ollama.py

import time
import requests
import numpy as np
import pandas as pd

INGREDIENT_CSV = "unique_ingredients.csv"   # 네가 만든 파일
OUT_META = "ingredient_meta.csv"
OUT_EMB = "ingredient_embeddings.npy"

OLLAMA_URL = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "nomic-embed-text"  # or bge-m3


def embed_text(text: str):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": text
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()
    return np.array(r.json()["embedding"], dtype=np.float32)


def main():
    df = pd.read_csv(INGREDIENT_CSV)

    # 컬럼명 자동 대응
    col = df.columns[0]
    ingredient_names = (
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    embeddings = []
    meta_rows = []

    print(f"[INFO] 총 성분 수: {len(ingredient_names)}")

    for i, ing in enumerate(ingredient_names):
        emb = embed_text(ing)
        embeddings.append(emb)

        meta_rows.append({
            "ingredient_id": i,
            "ingredient_name": ing
        })

        if (i + 1) % 50 == 0:
            print(f"[INFO] {i+1}/{len(ingredient_names)}")

        time.sleep(0.2)  # rate-limit 완화

    emb_array = np.stack(embeddings)

    pd.DataFrame(meta_rows).to_csv(OUT_META, index=False)
    np.save(OUT_EMB, emb_array)

    print("완료")
    print("ingredient_embeddings shape:", emb_array.shape)


if __name__ == "__main__":
    main()