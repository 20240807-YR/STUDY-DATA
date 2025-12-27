# ============================================================
# Brand Analysis Result → Reindexed Segmentation
# ============================================================

import pandas as pd
from pathlib import Path

# ----------------------------
# 경로 (12_27 고정)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

IN_PATH = DATA_DIR / "brand_analysis_result.csv"
OUT_PATH = DATA_DIR / "brand_analysis_segmented.csv"

# ----------------------------
# 로드
# ----------------------------
df = pd.read_csv(IN_PATH)

print("[INFO] columns:", list(df.columns))

# ----------------------------
# 컬럼 자동 탐색 (한글 대응)
# ----------------------------
brand_col = "브랜드" if "브랜드" in df.columns else "brand"
content_col = "내용 요약" if "내용 요약" in df.columns else "content"

# ----------------------------
# 브랜드별 part_id 재부여
# ----------------------------
rows = []

for brand, g in df.groupby(brand_col):
    g = g.reset_index(drop=True)

    for i, r in g.iterrows():
        text = str(r[content_col]).strip()
        if not text:
            continue

        rows.append({
            "brand": brand,
            "part_id": i + 1,              # ← 여기 핵심
            "content": text,
            "char_len": len(text)
        })

out_df = pd.DataFrame(rows)

# ----------------------------
# 저장
# ----------------------------
out_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("=" * 60)
print("brand_analysis_segmented.csv 재생성 완료")
print("ROWS:", len(out_df))
print("BRANDS:", out_df["brand"].nunique())
print("MAX part_id:", out_df["part_id"].max())
print("PATH:", OUT_PATH)
print("=" * 60)