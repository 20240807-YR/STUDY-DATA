# ============================================================
# Brand Tone 정의 확장 + 클러스터 매핑 생성
# - input  : 없음 (하드코딩 정의)
# - output :
#   data/brand_tone_definitions.csv
#   data/brand_tone_cluster.csv
# ============================================================

import pandas as pd
from pathlib import Path

BASE_DIR = Path("third_week/12_27")
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 1) Brand Tone Definitions (확장: 12개)
# ------------------------------------------------------------
tone_rows = [
    # Scientific 계열
    ("Clinical", "임상·수치 중심", "임상 데이터, 수치 기반 검증, 효능 재현성을 강조하는 톤"),
    ("Dermatological", "피부과 전문", "장벽·민감·저자극 관점에서 안전성과 적합성을 설명하는 톤"),
    ("Ingredient-Driven", "성분 메커니즘", "핵심 성분과 작용 메커니즘을 논리적으로 설명하는 톤"),

    # Emotional 계열
    ("Empathetic", "공감·위로", "일상과 피부 고민에 공감하며 심리적 장벽을 낮추는 톤"),
    ("Care-Oriented", "돌봄·보호", "지속적 사용과 피부 보호 이미지를 강조하는 톤"),
    ("Inspirational", "변화·자신감", "긍정적 변화와 자신감을 자극하는 동기 부여 톤"),

    # Luxury 계열
    ("Prestige", "권위·헤리티지", "브랜드 역사와 프리미엄 가치를 강조하는 톤"),
    ("Minimal-Luxury", "절제된 고급", "간결하고 정제된 표현으로 고급감을 전달하는 톤"),
    ("Sensory-Luxury", "감각적 경험", "텍스처·향·사용 경험을 중심으로 한 톤"),

    # Casual 계열
    ("Friendly", "친근한 대화", "부담 없이 쉽게 설명하는 톤"),
    ("Playful", "위트·트렌디", "밝고 가벼운 위트로 주목도를 높이는 톤"),
    ("Direct", "직설·핵심", "핵심만 빠르게 전달하고 행동을 유도하는 톤"),
]

tone_df = pd.DataFrame(
    tone_rows,
    columns=["tone_id", "description_preview", "full_description"]
)

tone_def_path = DATA_DIR / "brand_tone_definitions.csv"
tone_df.to_csv(tone_def_path, index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# 2) Brand Tone Cluster (기존 4 → 확장 매핑)
# ------------------------------------------------------------
cluster_rows = [
    ("Scientific", "Clinical"),
    ("Scientific", "Dermatological"),
    ("Scientific", "Ingredient-Driven"),

    ("Emotional", "Empathetic"),
    ("Emotional", "Care-Oriented"),
    ("Emotional", "Inspirational"),

    ("Luxury", "Prestige"),
    ("Luxury", "Minimal-Luxury"),
    ("Luxury", "Sensory-Luxury"),

    ("Casual", "Friendly"),
    ("Casual", "Playful"),
    ("Casual", "Direct"),
]

cluster_df = pd.DataFrame(
    cluster_rows,
    columns=["brand", "brand_tone_cluster"]
)

cluster_path = DATA_DIR / "brand_tone_cluster.csv"
cluster_df.to_csv(cluster_path, index=False, encoding="utf-8-sig")

print("[OK] brand tone definitions:", tone_def_path)
print("[OK] brand tone clusters:", cluster_path)
print("[INFO] definitions:", tone_df.shape, "clusters:", cluster_df.shape)