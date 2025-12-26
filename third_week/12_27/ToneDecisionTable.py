# ============================================================
# ToneDecisionTable (Extended) 생성 - FIXED
# ============================================================

import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# 경로 (12_27 기준 고정)
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

TONE_CLUSTER_PATH = DATA_DIR / "brand_tone_cluster.csv"
TONE_PROFILE_PATH = DATA_DIR / "tone_centroid_profile.csv"
OUT_PATH = DATA_DIR / "ToneDecisionTable_extended.csv"

tone_cluster_df = pd.read_csv(TONE_CLUSTER_PATH)
tone_profile_df = pd.read_csv(TONE_PROFILE_PATH)

# ------------------------------------------------------------
# 메시지 목적
# ------------------------------------------------------------
MESSAGE_GOALS = [
    "awareness",
    "brand_story",
    "conversion",
    "cta_push",
    "retention",
    "seasonal_event"
]

# ------------------------------------------------------------
# 🔑 핵심: cluster → 상위 tone 매핑
# ------------------------------------------------------------
CLUSTER_TO_TONE = {
    "Clinical": "Scientific",
    "Ingredient-Driven": "Scientific",
    "Dermatological": "Scientific",

    "Empathetic": "Emotional",
    "Care-Oriented": "Emotional",
    "Inspirational": "Emotional",

    "Luxury": "Luxury",
    "Prestige": "Luxury",
    "Minimal-Luxury": "Luxury",
    "Sensory-Luxury": "Luxury",

    "Friendly": "Casual",
    "Playful": "Casual",
    "Direct": "Casual"
}

# ------------------------------------------------------------
# 톤 프로파일 추론
# ------------------------------------------------------------
def infer_tone_profile(cluster_name, message_goal):
    if cluster_name in ["Clinical", "Ingredient-Driven", "Dermatological"]:
        return "clinical_explain" if message_goal != "cta_push" else "clinical_cta"

    if cluster_name in ["Luxury", "Prestige", "Minimal-Luxury", "Sensory-Luxury"]:
        return "heritage_story"

    if cluster_name in ["Empathetic", "Care-Oriented", "Inspirational"]:
        return "gentle_relationship"

    if cluster_name in ["Friendly", "Playful"]:
        return "trendy_hook"

    return "direct_cta"

def infer_emotion_intensity(cluster_name):
    if cluster_name in ["Empathetic", "Inspirational"]:
        return "high"
    if cluster_name in ["Friendly", "Playful"]:
        return "mid"
    return "low"

def infer_proof_requirement(cluster_name):
    if cluster_name in ["Clinical", "Ingredient-Driven", "Dermatological"]:
        return "data"
    if cluster_name in ["Luxury", "Prestige", "Minimal-Luxury", "Sensory-Luxury"]:
        return "story"
    return "review"

# ------------------------------------------------------------
# 테이블 생성
# ------------------------------------------------------------
rows = []

for _, row in tone_cluster_df.iterrows():
    cluster_name = row["brand"]
    cluster_id = row["brand_tone_cluster"]
    brand_position = row["brand_position"]

    # 상위 tone 결정
    super_tone = CLUSTER_TO_TONE.get(cluster_name, "Casual")

    centroid_row = tone_profile_df[
        tone_profile_df["tone_id"] == super_tone
    ]

    sentence_len_bucket = (
        centroid_row["sentence_len"].values[0]
        if not centroid_row.empty else "mid"
    )

    for goal in MESSAGE_GOALS:
        rows.append({
            "brand_tone_cluster": cluster_id,
            "brand_position": brand_position,
            "message_goal": goal,
            "tone_profile_id": infer_tone_profile(cluster_name, goal),
            "sentence_len_bucket": sentence_len_bucket,
            "emotion_intensity": infer_emotion_intensity(cluster_name),
            "proof_requirement": infer_proof_requirement(cluster_name),
            "description": f"{cluster_name} 톤 기반 {goal} 메시지 전략"
        })

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("=" * 60)
print("ToneDecisionTable_extended 생성 완료")
print("ROWS:", len(out_df))
print("PATH:", OUT_PATH)
print("=" * 60)