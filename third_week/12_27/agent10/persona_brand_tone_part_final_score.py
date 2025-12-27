# persona_brand_tone_part_final_score.py
# ============================================================
# Persona × Brand × Tone × Part Role Final Scoring
# (FINAL FIX – path-safe, submission-ready, persona_id-safe)
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

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0])


def parse_embedding_vector(val, dim: int) -> np.ndarray:
    """
    embedding_vector가
    - list/np.ndarray 이든
    - '0.1,0.2,...' 문자열이든
    - 'array([...])' 같은 문자열이든
    숫자만 robust하게 뽑아 dim으로 맞춤.
    """
    if isinstance(val, (list, np.ndarray)):
        arr = np.array(val, dtype=np.float32).ravel()
    else:
        s = "" if pd.isna(val) else str(val)
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if not nums:
            arr = np.zeros(dim, dtype=np.float32)
        else:
            arr = np.array(nums, dtype=np.float32).ravel()

    if arr.size < dim:
        arr = np.pad(arr, (0, dim - arr.size))
    elif arr.size > dim:
        arr = arr[:dim]

    return arr.astype(np.float32)


def normalize_brand(x) -> str:
    s = "" if pd.isna(x) else str(x)
    s = s.replace("\u00a0", " ")
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


# ============================================================
# 로드 + 검증
# ============================================================

required_files = [
    PERSONA_VEC_PATH,
    PERSONA_META_PATH,
    BRAND_PART_PATH,
    BRAND_TONE_CLUSTER_PATH,
    TONE_CENTROID_VEC_PATH,
]

for p in required_files:
    if not p.exists():
        raise FileNotFoundError(f"입력 파일 없음: {p}")

persona_vectors = np.load(PERSONA_VEC_PATH).astype(np.float32)
persona_meta = pd.read_csv(PERSONA_META_PATH)

brand_part_df = pd.read_csv(BRAND_PART_PATH)
brand_tone_df = pd.read_csv(BRAND_TONE_CLUSTER_PATH)
tone_centroid_vectors = np.load(TONE_CENTROID_VEC_PATH).astype(np.float32)

print("[INFO] persona_vectors:", persona_vectors.shape)
print("[INFO] persona_meta cols:", list(persona_meta.columns))
print("[INFO] brand_part_df cols:", list(brand_part_df.columns))
print("[INFO] brand_tone_df cols:", list(brand_tone_df.columns))
print("[INFO] tone_centroid_vectors:", tone_centroid_vectors.shape)

# persona_meta 필수 컬럼
if "persona_id" not in persona_meta.columns:
    raise ValueError("persona_meta_v2.csv에 persona_id 컬럼이 없습니다.")
persona_meta = persona_meta.set_index("persona_id")
persona_ids = persona_meta.index.astype(str).tolist()

# persona_meta ↔ persona_vectors 수 검증
if len(persona_ids) != persona_vectors.shape[0]:
    raise ValueError(
        f"persona_meta rows({len(persona_ids)}) != persona_vectors rows({persona_vectors.shape[0]})"
    )

# 차원 검증
if tone_centroid_vectors.ndim != 2:
    raise ValueError(f"tone_centroid_embeddings.npy shape 이상: {tone_centroid_vectors.shape}")

dim = int(tone_centroid_vectors.shape[1])
if int(persona_vectors.shape[1]) != dim:
    raise ValueError(f"persona_vectors dim({persona_vectors.shape[1]}) != centroid dim({dim})")

# brand_part 필수 컬럼
if "브랜드" in brand_part_df.columns and "brand" not in brand_part_df.columns:
    brand_part_df = brand_part_df.rename(columns={"브랜드": "brand"})

need_part_cols = {"brand", "part_role", "embedding_vector"}
missing_part = need_part_cols - set(brand_part_df.columns)
if missing_part:
    raise ValueError(f"brand_analysis_part_enhanced.csv 필수 컬럼 누락: {missing_part}")

# brand_tone 필수 컬럼
need_tone_cols = {"brand", "brand_tone_cluster"}
missing_tone = need_tone_cols - set(brand_tone_df.columns)
if missing_tone:
    raise ValueError(f"brand_tone_cluster_by_brand.csv 필수 컬럼 누락: {missing_tone}")


# ============================================================
# 브랜드 정규화 + 매핑
# ============================================================

brand_part_df["brand"] = brand_part_df["brand"].map(normalize_brand)
brand_tone_df["brand"] = brand_tone_df["brand"].map(normalize_brand)

# brand_tone_cluster int 변환
try:
    brand_tone_df["brand_tone_cluster"] = brand_tone_df["brand_tone_cluster"].astype(int)
except Exception as e:
    raise ValueError(f"brand_tone_cluster int 변환 실패: {e}")

brand_to_cluster = (
    brand_tone_df
    .set_index("brand")["brand_tone_cluster"]
    .to_dict()
)

matched = int(brand_part_df["brand"].isin(brand_to_cluster).sum())
print(f"[INFO] brand match rows: {matched} / {len(brand_part_df)}")

missing_brands = sorted(
    set(brand_part_df.loc[~brand_part_df["brand"].isin(brand_to_cluster), "brand"].tolist())
)
if missing_brands:
    print("[WARN] missing brands (sample up to 30):", missing_brands[:30])


# ============================================================
# part embedding 준비
# ============================================================

vec_list = []
zero_cnt = 0
for v in brand_part_df["embedding_vector"].tolist():
    arr = parse_embedding_vector(v, dim)
    if np.all(arr == 0):
        zero_cnt += 1
    vec_list.append(arr)

brand_part_vectors = np.vstack(vec_list).astype(np.float32)

print("[INFO] brand_part_vectors:", brand_part_vectors.shape)
print("[WARN] zero vectors:", zero_cnt)


# ============================================================
# 스코어링
# ============================================================

rows = []
TOPK = 50

for p_idx, persona_id in enumerate(persona_ids):
    p_vec = persona_vectors[p_idx]

    for i in range(len(brand_part_df)):
        brand = brand_part_df.at[i, "brand"]
        if brand not in brand_to_cluster:
            continue

        cluster = int(brand_to_cluster[brand])
        if cluster < 0 or cluster >= tone_centroid_vectors.shape[0]:
            continue

        part_role = str(brand_part_df.at[i, "part_role"]).strip()
        c_vec = brand_part_vectors[i]

        s_content = cosine(p_vec, c_vec)
        s_tone = cosine(p_vec, tone_centroid_vectors[cluster])

        score = (
            W_CONTENT * s_content
            + W_TONE_CENTROID * s_tone
            + W_PART_ROLE * float(PART_ROLE_WEIGHT.get(part_role, 0.85))
        )

        rows.append({
            "persona_id": persona_id,
            "brand": brand,
            "part_role": part_role,
            "part_id": int(brand_part_df.at[i, "part_id"]) if "part_id" in brand_part_df.columns and pd.notna(brand_part_df.at[i, "part_id"]) else None,
            "brand_tone_cluster": cluster,
            "s_content": float(s_content),
            "s_tone_centroid": float(s_tone),
            "part_bonus": float(PART_ROLE_WEIGHT.get(part_role, 0.85)),
            "score": float(score),
        })

out_df = pd.DataFrame(rows)

if out_df.empty:
    out_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print("=" * 60)
    print("[DONE] 결과 없음 (ROWS=0)")
    print("가능 원인:")
    print("1) brand_analysis_part_enhanced.csv brand 값이 brand_tone_cluster_by_brand.csv brand 값과 불일치")
    print("2) brand_tone_cluster 값이 centroid index(0..K-1) 범위를 벗어남")
    print("PATH:", OUT_PATH)
    print("=" * 60)
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