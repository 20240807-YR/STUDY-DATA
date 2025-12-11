"""
full_product_embedding_pipeline.py

전체 기능:
1) CSV 로드
2) 로 전성분 파싱 -> 지피티도,, 제미나이도,, 일단은 안 됨..
3) 전성분 정규화
4) Ollama로 개별 성분 임베딩
5) 성분 벡터 평균 pooling → ingredient_vector
6) 제품 메타텍스트 임베딩
7) 최종 product_vector = concat(product_text_emb, ingredient_emb)
8) 결과 저장
"""

import os
import json
import time
import requests
import numpy as np
import pandas as pd
from openai import OpenAI
#환경변수 -> VScode/터미널 가상 환경 만들어서
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

CSV_PATH = "amore_with_category.csv"

COL_PRODUCT = "상품명"
COL_BRAND = "brand"
COL_CATEGORY = "category"
COL_SUBCAT = "subcategory"
COL_INGR = "전성분"

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "nomic-embed-text"

OUT_NPY = "final_product_vectors.npy"
OUT_META = "final_product_meta.csv"

def parse_ingredients(text: str):
    prompt = f"""
    아래 전성분 문자열을 JSON 배열로 파싱하라.
    - %, (), 용량 제거
    - 반드시 ["성분1", "성분2", ...] 형태의 JSON array만 출력
    전성분:
    {text}
    """

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "JSON 배열만 출력하라."},
            {"role": "user", "content": prompt}
        ]
    )
    try:
        return json.loads(res.choices[0].message.content)
    except:
        return []


def normalize_ingredient(name: str) -> str:
    """성분 이름 통일 규칙 (간단 버전)"""
    name = name.strip().lower()

    replacements = {
        "ingredient": "",
        "(보습제)": "",
        "%": "",
        "  ": " "
    }

    for a, b in replacements.items():
        name = name.replace(a, b)

    return name.strip()


#올라마
def embed_text(text: str):
    payload = {"model": OLLAMA_MODEL, "prompt": text}

    r = requests.post(OLLAMA_EMBED_URL, json=payload)
    r.raise_for_status()

    return r.json()["embedding"]


def compute_ingredient_vector(ingredients: list[str]):
    """개별 성분 embedding → 평균 pooling"""
    vectors = []

    for ing in ingredients:
        v = embed_text(ing)
        vectors.append(v)

    if len(vectors) == 0:
        return np.zeros(1024, dtype=np.float32)

    return np.mean(np.array(vectors, dtype=np.float32), axis=0)

def build_product_text(row):
    brand = str(row.get(COL_BRAND, ""))
    name = str(row.get(COL_PRODUCT, ""))
    cat = str(row.get(COL_CATEGORY, ""))
    sub = str(row.get(COL_SUBCAT, ""))

    return f"브랜드: {brand}. 상품명: {name}. 카테고리: {cat}. 서브카테고리: {sub}."


def concat_vectors(a: np.ndarray, b: np.ndarray):
    """두 벡터를 이어 붙임"""
    return np.concatenate([a, b], axis=0)


def main():
    df = pd.read_csv(CSV_PATH)
    product_vectors = []
    meta_rows = []

    total = len(df)

    for i, row in df.iterrows():
        print(f"[{i+1}/{total}] 처리 중…")

        raw_ingr = str(row.get(COL_INGR, ""))
        parsed_list = parse_ingredients(raw_ingr)

        # 터질까봐 Sleep
        time.sleep(0.4)

        normalized = [normalize_ingredient(x) for x in parsed_list]

        ingr_vec = compute_ingredient_vector(normalized)

        ptext = build_product_text(row)
        ptext_vec = np.array(embed_text(ptext), dtype=np.float32)

        final_vec = concat_vectors(ptext_vec, ingr_vec)
        product_vectors.append(final_vec)

        meta_rows.append([
            row.get(COL_BRAND, ""),
            row.get(COL_PRODUCT, ""),
            row.get(COL_CATEGORY, ""),
            row.get(COL_SUBCAT, "")
        ])

    # numpy 저장
    product_vectors = np.array(product_vectors, dtype=np.float32)
    np.save(OUT_NPY, product_vectors)

    meta_df = pd.DataFrame(meta_rows, columns=[
        COL_BRAND, COL_PRODUCT, COL_CATEGORY, COL_SUBCAT
    ])
    meta_df.to_csv(OUT_META, index=False)

    print("완료!")
    print("벡터 shape:", product_vectors.shape)
    print("저장:", OUT_NPY, OUT_META)


if __name__ == "__main__":
    main()