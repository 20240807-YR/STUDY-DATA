# amore_embedding.py
# 아모레 제품 CSV -> Ollama 임베딩 벡터 생성 스크립트

import os
import json
import time
import requests
import numpy as np
import pandas as pd


CSV_PATH = "amore_with_category.csv"

COL_NAME_PRODUCT   = "상품명"
COL_NAME_URL       = "URL"
COL_NAME_PRICE     = "price_original"
COL_NAME_INGR      = "전성분"
COL_NAME_CATEGORY  = "category"
COL_NAME_SUBCAT    = "subcategory"
COL_NAME_BRAND     = "brand"   # 추가된 브랜드 컬럼

EMBED_MODEL = "nomic-embed-text"

EMBED_NPY_PATH  = "amore_product_embeddings.npy"
META_CSV_PATH   = "amore_product_meta.csv"


def embed_text(text: str, model: str = EMBED_MODEL, retry: int = 3):
    url = "http://localhost:11434/api/embeddings"
    payload = {"model": model, "prompt": text}

    for attempt in range(1, retry + 1):
        try:
            r = requests.post(url, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            emb = data.get("embedding")
            return emb
        except Exception as e:
            print(f"[embed_text] 시도 {attempt}/{retry} 실패: {e}")
            if attempt == retry:
                return None
            time.sleep(1.0)


def build_product_text(row: pd.Series) -> str:
    name  = str(row.get(COL_NAME_PRODUCT, "") or "")
    brand = str(row.get(COL_NAME_BRAND, "") or "")
    cat   = str(row.get(COL_NAME_CATEGORY, "") or "")
    sub   = str(row.get(COL_NAME_SUBCAT, "") or "")
    ingr  = str(row.get(COL_NAME_INGR, "") or "")

    text = (
        f"브랜드: {brand}. "
        f"상품명: {name}. "
        f"카테고리: {cat}. "
        f"서브카테고리: {sub}. "
        f"전성분: {ingr}"
    )
    return text


def main():
    print(f"[INFO] CSV 읽는 중: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    df = df.copy()
    for col in [COL_NAME_PRODUCT, COL_NAME_BRAND, COL_NAME_CATEGORY, COL_NAME_SUBCAT, COL_NAME_INGR]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    num_rows = len(df)
    print(f"[INFO] 총 {num_rows}개 제품")

    embeddings = []
    valid_idx = []

    for idx, row in df.iterrows():
        text = build_product_text(row)

        if not text.strip():
            embeddings.append(None)
            continue

        emb = embed_text(text)
        if emb is None:
            embeddings.append(None)
            continue

        embeddings.append(emb)
        valid_idx.append(idx)

        if (idx + 1) % 50 == 0 or idx == num_rows - 1:
            print(f"[INFO] 진행률: {idx + 1}/{num_rows}")

    valid_embeddings = [emb for emb in embeddings if emb is not None]

    if not valid_embeddings:
        print("[ERROR] 유효한 임베딩이 하나도 없음. 종료.")
        return

    emb_array = np.array(valid_embeddings, dtype=np.float32)
    print(f"[INFO] 임베딩 배열 shape: {emb_array.shape}")

    np.save(EMBED_NPY_PATH, emb_array)
    print(f"[INFO] 임베딩 저장 완료: {EMBED_NPY_PATH}")

    meta_df = df.loc[valid_idx, [
        COL_NAME_PRODUCT,
        COL_NAME_BRAND,
        COL_NAME_URL,
        COL_NAME_PRICE,
        COL_NAME_CATEGORY,
        COL_NAME_SUBCAT
    ]].reset_index(drop=True)

    meta_df.to_csv(META_CSV_PATH, index=False)
    print(f"[INFO] 메타 정보 저장 완료: {META_CSV_PATH}")


if __name__ == "__main__":
    main()