# ============================================================
# Persona × Product FULL Similarity 생성
# (현재 12_27/data 구조 기준)
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# 경로 설정
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PERSONA_VEC_PATH = DATA_DIR / "persona_vectors.npy"
PRODUCT_VEC_PATH = DATA_DIR / "product_embeddings.npy"
PRODUCT_META_PATH = DATA_DIR / "amore_product_meta.csv"

OUT_PATH = DATA_DIR / "persona_product_similarity_full.csv"

# ----------------------------
# 로드
# ----------------------------
persona_vecs = np.load(PERSONA_VEC_PATH)        # (P, D)
product_vecs = np.load(PRODUCT_VEC_PATH)        # (N, D)
product_meta = pd.read_csv(PRODUCT_META_PATH)

print("[INFO] persona vectors:", persona_vecs.shape)
print("[INFO] product vectors:", product_vecs.shape)
print("[INFO] product meta:", product_meta.shape)

# ----------------------------
# cosine similarity
# ----------------------------
sim_matrix = cosine_similarity(persona_vecs, product_vecs)  # (P, N)

# ----------------------------
# long-format CSV 생성
# ----------------------------
rows = []

for p_idx in range(sim_matrix.shape[0]):
    persona_id = f"persona_{p_idx+1}"
    for prod_idx in range(sim_matrix.shape[1]):
        rows.append({
            "persona_id": persona_id,
            "product_index": prod_idx,
            "similarity": float(sim_matrix[p_idx, prod_idx]),
            "brand": product_meta.loc[prod_idx, "brand"],
            "product_name": product_meta.loc[prod_idx, "상품명"],
            "category": product_meta.loc[prod_idx, "category"],
            "subcategory": product_meta.loc[prod_idx, "subcategory"],
        })

df_out = pd.DataFrame(rows)

df_out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("=" * 50)
print("FULL similarity CSV 생성 완료")
print("PATH:", OUT_PATH)
print("ROWS:", len(df_out))
print("=" * 50)