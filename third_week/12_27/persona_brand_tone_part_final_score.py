# persona_brand_tone_part_final_score.py
# ============================================================
# Persona × Brand × Tone × Part Role Final Scoring (debug + robust brand match)
# - ONLY: third_week/12_27/data/*
# - fixes:
#   1) embedding_vector robust parse
#   2) brand normalization + alias mapping
#   3) diagnostics: how many brands matched / missing examples
# ============================================================

import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 경로 (third_week/12_27/ 에서 실행)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PERSONA_VEC_PATH = DATA_DIR / "persona_vectors.npy"
PERSONA_META_PATH = DATA_DIR / "persona_meta_v2.csv"

BRAND_PART_PATH = DATA_DIR / "brand_analysis_part_enhanced.csv"
BRAND_TONE_CLUSTER_PATH = DATA_DIR / "brand_tone_cluster.csv"

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
    "cta": 0.85
}


# ============================================================
# 유틸
# ============================================================

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0])


def _parse_embedding_vector(val, expected_dim: int) -> np.ndarray:
    if isinstance(val, np.ndarray):
        arr = val.astype(np.float32).ravel()
    elif isinstance(val, list):
        arr = np.array(val, dtype=np.float32).ravel()
    else:
        s = "" if pd.isna(val) else str(val)
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if not nums:
            return np.zeros(expected_dim, dtype=np.float32)
        arr = np.array(nums, dtype=np.float32).ravel()

    if arr.size < expected_dim:
        arr = np.concatenate([arr, np.zeros(expected_dim - arr.size, dtype=np.float32)], axis=0)
    elif arr.size > expected_dim:
        arr = arr[:expected_dim]

    return arr.astype(np.float32)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    if "브랜드" in df.columns and "brand" not in df.columns:
        rename_map["브랜드"] = "brand"
    if "내용 요약" in df.columns and "content" not in df.columns:
        rename_map["내용 요약"] = "content"
    if "구분" in df.columns and "part_role" not in df.columns:
        rename_map["구분"] = "part_role"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _norm_brand(x: str) -> str:
    """
    브랜드 키 정규화:
    - strip
    - NBSP 제거
    - 여러 공백 -> 1칸
    - 특수문자(슬래시/밑줄 등) 양끝 공백 제거
    """
    s = "" if pd.isna(x) else str(x)
    s = s.replace("\u00a0", " ")  # NBSP
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _build_brand_alias_map(source_brands, target_brands):
    """
    최소한의 별칭 매핑(필요하면 여기에 계속 추가)
    - source: brand_part_df 쪽
    - target: brand_tone_df 쪽
    """
    # 기본: 정규화 동일하면 그대로
    alias = {}

    # 흔한 케이스: 영문 약칭/공백/대소문자 차이
    # (현재 스크린샷/샘플 기준으로 'AP'가 자주 등장)
    if "AP" in source_brands:
        # target에 'AP'가 없고 유사한게 있으면 매핑
        # 예: "A P", "A.P", "APEX" 같은 식
        candidates = [t for t in target_brands if _norm_brand(t).replace(".", "").replace(" ", "").upper() == "AP"]
        if candidates:
            alias["AP"] = candidates[0]

    # 필요하면 여기에 직접 추가
    # alias["아윤채"] = "아윤채"  # 예시
    return alias


# ============================================================
# 로드 + 검증
# ============================================================

for p in [PERSONA_VEC_PATH, PERSONA_META_PATH, BRAND_PART_PATH, BRAND_TONE_CLUSTER_PATH, TONE_CENTROID_VEC_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"입력 파일 없음: {p}")

persona_vectors = np.load(PERSONA_VEC_PATH)
persona_meta = pd.read_csv(PERSONA_META_PATH)

brand_part_df = pd.read_csv(BRAND_PART_PATH)
brand_tone_df = pd.read_csv(BRAND_TONE_CLUSTER_PATH)

tone_centroid_vectors = np.load(TONE_CENTROID_VEC_PATH)

print("[INFO] persona_vectors:", persona_vectors.shape)
print("[INFO] persona_meta columns:", list(persona_meta.columns))
print("[INFO] brand_part_df columns:", list(brand_part_df.columns))
print("[INFO] brand_tone_df columns:", list(brand_tone_df.columns))
print("[INFO] tone_centroid_vectors:", tone_centroid_vectors.shape)

brand_part_df = _normalize_columns(brand_part_df)

if "persona_id" not in persona_meta.columns:
    raise ValueError(f"persona_meta_v2.csv에 persona_id 없음. columns={list(persona_meta.columns)}")
persona_meta = persona_meta.set_index("persona_id")

required_part_cols = {"brand", "part_role", "embedding_vector"}
missing = required_part_cols - set(brand_part_df.columns)
if missing:
    raise ValueError(f"brand_analysis_part_enhanced.csv 필수 컬럼 누락: {missing} | columns={list(brand_part_df.columns)}")

required_tone_cols = {"brand", "brand_tone_cluster"}
missing = required_tone_cols - set(brand_tone_df.columns)
if missing:
    raise ValueError(f"brand_tone_cluster.csv 필수 컬럼 누락: {missing} | columns={list(brand_tone_df.columns)}")

if tone_centroid_vectors.ndim != 2:
    raise ValueError(f"tone_centroid_embeddings.npy shape 이상: {tone_centroid_vectors.shape}")

dim = tone_centroid_vectors.shape[1]
if persona_vectors.shape[1] != dim:
    raise ValueError(f"persona_vectors dim({persona_vectors.shape[1]}) != tone_centroid_vectors dim({dim})")

# ============================================================
# brand 정규화 + alias 매핑
# ============================================================

brand_part_df["brand_raw"] = brand_part_df["brand"]
brand_tone_df["brand_raw"] = brand_tone_df["brand"]

brand_part_df["brand"] = brand_part_df["brand"].map(_norm_brand)
brand_tone_df["brand"] = brand_tone_df["brand"].map(_norm_brand)

source_brands = set(brand_part_df["brand"].unique().tolist())
target_brands = set(brand_tone_df["brand"].unique().tolist())

alias_map = _build_brand_alias_map(source_brands, target_brands)

# alias 적용
if alias_map:
    brand_part_df["brand"] = brand_part_df["brand"].replace(alias_map)

# 매칭 진단
matched = brand_part_df["brand"].isin(target_brands).sum()
total = len(brand_part_df)
unique_missing = sorted(list(set(brand_part_df.loc[~brand_part_df["brand"].isin(target_brands), "brand"].unique().tolist())))
print("[INFO] brand match rows:", matched, "/", total)
print("[INFO] unique brand_part:", len(set(brand_part_df["brand"].unique())))
print("[INFO] unique brand_tone:", len(set(brand_tone_df["brand"].unique())))
if unique_missing:
    print("[WARN] missing brands (sample up to 20):", unique_missing[:20])
else:
    print("[INFO] all brands matched")

# ============================================================
# brand_part 임베딩 벡터 준비
# ============================================================

vecs = []
zero_cnt = 0
for v in brand_part_df["embedding_vector"].tolist():
    arr = _parse_embedding_vector(v, expected_dim=dim)
    if np.all(arr == 0):
        zero_cnt += 1
    vecs.append(arr)
brand_part_vectors = np.vstack(vecs).astype(np.float32)

print("[INFO] brand_part_vectors:", brand_part_vectors.shape)
print("[WARN] zero vectors:", zero_cnt)

# ============================================================
# 매핑
# ============================================================

cols = ["brand_tone_cluster"]
if "brand_position" in brand_tone_df.columns:
    cols.append("brand_position")

brand_to_tone = (
    brand_tone_df
    .set_index("brand")[cols]
    .to_dict(orient="index")
)

# ============================================================
# 스코어링
# ============================================================

rows = []
TOPK = 50

skipped_no_brand = 0
skipped_bad_cluster = 0

for persona_idx in range(persona_vectors.shape[0]):
    persona_id = f"persona_{persona_idx+1}"
    persona_vec = persona_vectors[persona_idx].astype(np.float32)

    for i in range(len(brand_part_df)):
        brand = brand_part_df.at[i, "brand"]
        part_role = str(brand_part_df.at[i, "part_role"]).strip()

        if brand not in brand_to_tone:
            skipped_no_brand += 1
            continue

        tone_cluster = int(brand_to_tone[brand]["brand_tone_cluster"])
        if tone_cluster < 0 or tone_cluster >= tone_centroid_vectors.shape[0]:
            skipped_bad_cluster += 1
            continue

        content_vec = brand_part_vectors[i]

        s_content = _cos(persona_vec, content_vec)
        s_tone_centroid = _cos(persona_vec, tone_centroid_vectors[tone_cluster].astype(np.float32))

        part_bonus = PART_ROLE_WEIGHT.get(part_role, 0.85)

        final_score = (
            W_CONTENT * s_content +
            W_TONE_CENTROID * s_tone_centroid +
            W_PART_ROLE * part_bonus
        )

        rows.append({
            "persona_id": persona_id,
            "brand": brand,
            "part_role": part_role,
            "part_id": int(brand_part_df.at[i, "part_id"]) if "part_id" in brand_part_df.columns and pd.notna(brand_part_df.at[i, "part_id"]) else None,
            "brand_tone_cluster": tone_cluster,
            "s_content": float(s_content),
            "s_tone_centroid": float(s_tone_centroid),
            "part_bonus": float(part_bonus),
            "score": float(final_score),
        })

out_df = pd.DataFrame(rows)

print("[INFO] skipped_no_brand:", skipped_no_brand)
print("[INFO] skipped_bad_cluster:", skipped_bad_cluster)

if out_df.empty:
    out_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print("=" * 60)
    print("[DONE] 결과 없음 (ROWS=0)")
    print("원인 후보:")
    print("1) brand_analysis_part_enhanced.csv의 brand 값과 brand_tone_cluster.csv의 brand 값이 불일치")
    print("2) brand_tone_cluster.csv의 brand_tone_cluster가 centroid index와 불일치(0..N-1 아님)")
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