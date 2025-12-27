# ============================================================
# Persona × Brand × Tone × Part Role Final Scoring
# (FINAL FIX – path-safe, submission-ready)
# ============================================================

import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 🔒 경로 기준 (절대 안전)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent        # third_week/12_27
DATA_DIR = BASE_DIR / "data"                      # third_week/12_27/data

PERSONA_VEC_PATH = DATA_DIR / "persona_vectors.npy"
PERSONA_META_PATH = DATA_DIR / "persona_meta_v2.csv"

BRAND_PART_PATH = DATA_DIR / "brand_analysis_part_enhanced.csv"
BRAND_TONE_CLUSTER_PATH = DATA_DIR / "brand_tone_cluster_by_brand.csv"

TONE_CENTROID_VEC_PATH = DATA_DIR / "tone_centroid_embeddings.npy"

OUT_PATH = DATA_DIR / "persona_brand_tone_part_final.csv"


# ============================================================
# 가중치
# ============================================================

W_CONTENT = 0.70
W_TONE_CENTROID = 0.25
W_PART_ROLE = 0.05

PART_ROLE_WEIGHT = {
    "identity": 1.00,
    "benefit": 0.95,
    "proof": 0.92,
    "emotion": 0.90,
    "cta": 0.85,
}


# ============================================================
# 유틸
# ============================================================

def cosine(a, b):
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0])


def parse_embedding_vector(val, dim):
    if isinstance(val, (list, np.ndarray)):
        arr = np.array(val, dtype=np.float32).ravel()
    else:
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(val))
        arr = np.array(nums, dtype=np.float32) if nums else np.zeros(dim, dtype=np.float32)

    if arr.size < dim:
        arr = np.pad(arr, (0, dim - arr.size))
    elif arr.size > dim:
        arr = arr[:dim]

    return arr.astype(np.float32)


def normalize_brand(x):
    s = "" if pd.isna(x) else str(x)
    s = s.replace("\u00a0", " ")
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


# ============================================================
# 로드 + 검증
# ============================================================

for p in [
    PERSONA_VEC_PATH,
    PERSONA_META_PATH,
    BRAND_PART_PATH,
    BRAND_TONE_CLUSTER_PATH,
    TONE_CENTROID_VEC_PATH,
]:
    if not p.exists():
        raise FileNotFoundError(f"입력 파일 없음: {p}")

persona_vectors = np.load(PERSONA_VEC_PATH).astype(np.float32)
persona_meta = pd.read_csv(PERSONA_META_PATH).set_index("persona_id")

brand_part_df = pd.read_csv(BRAND_PART_PATH)
brand_tone_df = pd.read_csv(BRAND_TONE_CLUSTER_PATH)
tone_centroid_vectors = np.load(TONE_CENTROID_VEC_PATH).astype(np.float32)

print("[INFO] persona_vectors:", persona_vectors.shape)
print("[INFO] brand_part_df cols:", list(brand_part_df.columns))
print("[INFO] brand_tone_df cols:", list(brand_tone_df.columns))
print("[INFO] tone_centroid_vectors:", tone_centroid_vectors.shape)


# ============================================================
# 컬럼 / 브랜드 정리
# ============================================================

if "브랜드" in brand_part_df.columns and "brand" not in brand_part_df.columns:
    brand_part_df = brand_part_df.rename(columns={"브랜드": "brand"})

brand_part_df["brand"] = brand_part_df["brand"].map(normalize_brand)
brand_tone_df["brand"] = brand_tone_df["brand"].map(normalize_brand)

brand_to_cluster = (
    brand_tone_df
    .set_index("brand")["brand_tone_cluster"]
    .astype(int)
    .to_dict()
)

matched = brand_part_df["brand"].isin(brand_to_cluster).sum()
print(f"[INFO] brand match rows: {matched} / {len(brand_part_df)}")


# ============================================================
# part embedding 준비
# ============================================================

dim = tone_centroid_vectors.shape[1]
brand_part_vectors = np.vstack([
    parse_embedding_vector(v, dim)
    for v in brand_part_df["embedding_vector"]
]).astype(np.float32)

print("[INFO] brand_part_vectors:", brand_part_vectors.shape)


# ============================================================
# 스코어링
# ============================================================

rows = []
TOPK = 50

for p_idx in range(persona_vectors.shape[0]):
    persona_id = f"persona_{p_idx+1}"
    p_vec = persona_vectors[p_idx]

    for i in range(len(brand_part_df)):
        brand = brand_part_df.at[i, "brand"]
        if brand not in brand_to_cluster:
            continue

        cluster = brand_to_cluster[brand]
        if cluster < 0 or cluster >= tone_centroid_vectors.shape[0]:
            continue

        part_role = str(brand_part_df.at[i, "part_role"]).strip()
        c_vec = brand_part_vectors[i]

        score = (
            W_CONTENT * cosine(p_vec, c_vec)
            + W_TONE_CENTROID * cosine(p_vec, tone_centroid_vectors[cluster])
            + W_PART_ROLE * PART_ROLE_WEIGHT.get(part_role, 0.85)
        )

        rows.append({
            "persona_id": persona_id,
            "brand": brand,
            "part_role": part_role,
            "part_id": brand_part_df.at[i, "part_id"] if "part_id" in brand_part_df.columns else None,
            "brand_tone_cluster": cluster,
            "score": float(score),
        })


out_df = pd.DataFrame(rows)

if out_df.empty:
    out_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print("[DONE] 결과 없음")
    raise SystemExit(0)

out_df = (
    out_df.sort_values("score", ascending=False)
          .groupby("persona_id", as_index=False)
          .head(TOPK)
          .reset_index(drop=True)
)

out_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("=" * 60)
print("persona × brand × tone × part_role FINAL SCORE 완료")
print("ROWS:", len(out_df))
print("TOPK per persona:", TOPK)
print("PATH:", OUT_PATH)
print("=" * 60)