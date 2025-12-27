# ============================================================
# Persona × Brand × Tone × Part Role Final Scoring
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 경로 설정 (고정)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PERSONA_VEC_PATH = DATA_DIR / "persona_vectors.npy"
PERSONA_META_PATH = DATA_DIR / "persona_meta_v2.csv"

BRAND_PART_PATH = DATA_DIR / "brand_analysis_part_enhanced.csv"
BRAND_TONE_CLUSTER_PATH = DATA_DIR / "brand_tone_cluster.csv"

TONE_CENTROID_VEC_PATH = DATA_DIR / "tone_centroid_embeddings.npy"
TONE_DECISION_PATH = DATA_DIR / "ToneDecisionTable_extended.csv"
TONE_DECISION_VEC_PATH = DATA_DIR / "tone_decision_embeddings.npy"

OUT_PATH = DATA_DIR / "persona_brand_tone_part_final.csv"

# ============================================================
# 가중치 (조정 가능)
# ============================================================

W_CONTENT = 0.45
W_TONE_CENTROID = 0.25
W_TONE_DECISION = 0.20
W_PART_ROLE = 0.05
W_BRAND_POSITION = 0.05

PART_ROLE_WEIGHT = {
    "identity": 1.0,
    "benefit": 0.9,
    "proof": 0.85,
    "emotion": 0.8,
    "cta": 0.7
}

# ============================================================
# 유틸
# ============================================================

def cosine(a, b):
    return cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0]

def brand_persona_modifier(brand_position, persona_type):
    if brand_position == "고기능/클리니컬 톤" and persona_type == "emotional":
        return 0.9
    if brand_position == "트렌디/컬러 중심 톤" and persona_type == "clinical":
        return 0.9
    return 1.0

# ============================================================
# 로드
# ============================================================

persona_vectors = np.load(PERSONA_VEC_PATH)
persona_meta = pd.read_csv(PERSONA_META_PATH).set_index("persona_id")

brand_part_df = pd.read_csv(BRAND_PART_PATH)
brand_tone_df = pd.read_csv(BRAND_TONE_CLUSTER_PATH)

tone_centroid_vectors = np.load(TONE_CENTROID_VEC_PATH)

tone_decision_df = pd.read_csv(TONE_DECISION_PATH)
tone_decision_vectors = np.load(TONE_DECISION_VEC_PATH)

# ============================================================
# 매핑 구성
# ============================================================

# brand → tone_cluster / brand_position
brand_to_tone = (
    brand_tone_df
    .set_index("brand")[["brand_tone_cluster", "brand_position"]]
    .to_dict(orient="index")
)

# tone_cluster → centroid vector
tone_centroid_map = {
    i: tone_centroid_vectors[i]
    for i in range(len(tone_centroid_vectors))
}

# tone decision id → vector
tone_decision_map = {
    i: tone_decision_vectors[i]
    for i in range(len(tone_decision_vectors))
}

# ============================================================
# 메인 스코어 루프
# ============================================================

rows = []

for persona_idx, persona_vec in enumerate(persona_vectors):
    persona_id = f"persona_{persona_idx+1}"
    persona_type = persona_meta.loc[persona_id, "persona_type"]

    for i, row in brand_part_df.iterrows():
        brand = row["brand"]
        part_role = row["part_role"]
        content_vec = np.fromstring(row["embedding_vector"], sep=",")

        if brand not in brand_to_tone:
            continue

        tone_cluster = int(brand_to_tone[brand]["brand_tone_cluster"])
        brand_position = brand_to_tone[brand]["brand_position"]

        # tone decision: cluster + part_role 기준 첫 매칭
        decision_rows = tone_decision_df[
            (tone_decision_df["brand_tone_cluster"] == tone_cluster) &
            (tone_decision_df["part_role"] == part_role)
        ]

        if decision_rows.empty:
            continue

        decision_idx = decision_rows.index[0]
        tone_decision_vec = tone_decision_map[decision_idx]

        # ----------------------------
        # score 계산
        # ----------------------------

        s_content = cosine(persona_vec, content_vec)
        s_tone_centroid = cosine(persona_vec, tone_centroid_map[tone_cluster])
        s_tone_decision = cosine(persona_vec, tone_decision_vec)

        part_bonus = PART_ROLE_WEIGHT.get(part_role, 0.7)
        brand_bonus = brand_persona_modifier(brand_position, persona_type)

        final_score = (
            W_CONTENT * s_content +
            W_TONE_CENTROID * s_tone_centroid +
            W_TONE_DECISION * s_tone_decision +
            W_PART_ROLE * part_bonus
        ) * brand_bonus

        rows.append({
            "persona_id": persona_id,
            "brand": brand,
            "part_role": part_role,
            "brand_tone_cluster": tone_cluster,
            "score": float(final_score)
        })

# ============================================================
# 결과 정리
# ============================================================

out_df = (
    pd.DataFrame(rows)
    .sort_values("score", ascending=False)
    .groupby("persona_id")
    .head(10)
    .reset_index(drop=True)
)

out_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("=" * 60)
print("persona × brand × tone × part_role FINAL SCORE 완료")
print("ROWS:", len(out_df))
print("PATH:", OUT_PATH)
print("=" * 60)