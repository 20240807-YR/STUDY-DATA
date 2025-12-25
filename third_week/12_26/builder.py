# builder.py
from schemas import CRMInput

def build_crm_input(row, tone_vector) -> CRMInput:
    return {
        "persona_id": row["persona_id"],
        "brand": row["brand"],
        "brand_tone_cluster": int(row["brand_tone_cluster"]),
        "avg_similarity": float(row["avg_similarity"]),
        # ingredient_affinity 제거 (이 CSV에는 없음)
        "tone_vector": tone_vector,
        "product_cnt": int(row["product_cnt"]),
    }