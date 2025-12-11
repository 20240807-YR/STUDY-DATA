"""
full_product_embedding_pipeline_ollama.py

전체 기능:
1) CSV 로드
2) Gemini로 전성분 파싱 (LLM reasoning 필요)
3) 전성분 정규화
4) Ollama로 개별 성분 임베딩
5) Ollama로 제품 메타텍스트 임베딩
6) 성분/제품 벡터 concat
7) 결과 저장

Ollama 모델 예: "nomic-embed-text" 또는 "bge-m3" (로컬 설치된 임베딩 모델)
"""

import os
import json
import time
import requests
import numpy as np
import pandas as pd
import google.generativeai as genai


# -----------------------------
# 0. Gemini API 설정
# -----------------------------
# 반드시 GEMINI_API_KEY 환경변수 세팅할 것
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-1.5-flash")


# -----------------------------
# 1. 설정값
# -----------------------------
CSV_PATH = "amore_with_category.csv"

COL_PRODUCT = "상품명"
COL_BRAND = "brand"
COL_CATEGORY = "category"
COL_SUBCAT = "subcategory"
COL_INGR = "전성분"

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "nomic-embed-text"  # 또는 bge-m3 등 너가 설치한 임베딩 모델 이름

OUT_NPY = "final_product_vectors.npy"
OUT_META = "final_product_meta.csv"


# -----------------------------
# 2. Ollama 임베딩
# -----------------------------
def embed_text(text: str):
    """Ollama 로컬 임베딩"""
    if not text:
        return np.zeros(1024, dtype=np.float32)

    payload = {"model": OLLAMA_MODEL, "prompt": text}
    r = requests.post(OLLAMA_EMBED_URL, json=payload)
    r.raise_for_status()

    emb = r.json()["embedding"]
    return np.array(emb, dtype=np.float32)


# -----------------------------
# 3. Gemini 전성분 파싱
# -----------------------------
def parse_ingredients(text: str):
    prompt = f"""
아래 화장품 전성분 문자열을 JSON 배열로 파싱하라.

규칙:
- %, (), 숫자, 용량 제거
- 성분명만 남기기
- 반드시 ["성분1", "성분2", ...] 형태의 JSON만 출력
- JSON 외 텍스트 금지

전성분:
{text}
"""

    try:
        res = gemini_model.generate_content(prompt)
        content = res.text.strip()
        return json.loads(content)
    except:
        return []


# -----------------------------
# 4. 전성분 정규화
# -----------------------------
def normalize_ingredient(name: str) -> str:
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


# -----------------------------
# 5. Ingredient Embedding (평균 pooling)
# -----------------------------
def compute_ingredient_vector(ingredients: list[str]):
    ingredients = [x for x in ingredients if x.strip()]
    if not ingredients:
        return np.zeros(1024, dtype=np.float32)

    vecs = [embed_text(ing) for ing in ingredients]
    return np.mean(np.array(vecs, dtype=np.float32), axis=0)


# -----------------------------
# 6. 메타 텍스트 생성
# -----------------------------
def build_product_text(row):
    brand = str(row.get(COL_BRAND, "")).strip()
    name = str(row.get(COL_PRODUCT, "")).strip()
    cat = str(row.get(COL_CATEGORY, "")).strip()
    sub = str(row.get(COL_SUBCAT, "")).strip()

    return f"브랜드: {brand}. 상품명: {name}. 카테고리: {cat}. 서브카테고리: {sub}."


# -----------------------------
# 7. 벡터 concat
# -----------------------------
def concat_vectors(a, b):
    return np.concatenate([a, b], axis=0)


# -----------------------------
# 8. 전체 실행
# -----------------------------
def main():
    df = pd.read_csv(CSV_PATH)
    product_vectors = []
    meta_rows = []
    total = len(df)

    for i, row in df.iterrows():
        print(f"[{i+1}/{total}] 처리 중…")

        # (1) Gemini로 성분 파싱
        raw_ingr = str(row.get(COL_INGR, ""))
        parsed = parse_ingredients(raw_ingr)
        time.sleep(0.4)  # rate-limit 완화

        # (2) 정규화
        normalized = [normalize_ingredient(x) for x in parsed]

        # (3) 성분 임베딩
        ingr_vec = compute_ingredient_vector(normalized)

        # (4) 제품 메타텍스트 임베딩
        ptext = build_product_text(row)
        ptext_vec = embed_text(ptext)

        # (5) concat → 최종 벡터
        final_vec = concat_vectors(ptext_vec, ingr_vec)
        product_vectors.append(final_vec)

        # meta 저장
        meta_rows.append([
            row.get(COL_BRAND, ""),
            row.get(COL_PRODUCT, ""),
            row.get(COL_CATEGORY, ""),
            row.get(COL_SUBCAT, "")
        ])

    # 저장
    product_vectors = np.array(product_vectors, dtype=np.float32)
    np.save(OUT_NPY, product_vectors)

    meta_df = pd.DataFrame(meta_rows, columns=[
        COL_BRAND, COL_PRODUCT, COL_CATEGORY, COL_SUBCAT
    ])
    meta_df.to_csv(OUT_META, index=False)

    print("완료!")
    print("shape:", product_vectors.shape)


if __name__ == "__main__":
    main()