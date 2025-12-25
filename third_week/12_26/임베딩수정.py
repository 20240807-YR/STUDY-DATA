# =============================================================================
# Persona × Product Similarity + Persona × Brand × Tone Summary (AUTO ALIGN)
# =============================================================================
import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------------------------------------------------------
# PATHS (절대경로 안전)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1] / "data_csv"

P_PATH    = BASE_DIR / "persona_vectors.npy"
X_PATH    = BASE_DIR / "final_product_with_brandtone.npy"
META_PATH = BASE_DIR / "final_product_with_brandtone_meta.csv"
SEG_PATH  = BASE_DIR / "final_brand_segments.csv"

OUT_FULL = BASE_DIR / "persona_product_similarity_full.csv"
OUT_TOPN = BASE_DIR / "persona_product_similarity_topN.csv"
OUT_STR  = BASE_DIR / "persona_brand_tone_strategy.csv"

TOPN = 50

# -----------------------------------------------------------------------------
# 1) LOAD
# -----------------------------------------------------------------------------
P = np.load(P_PATH).astype(np.float32)   # (P, Dp)
X = np.load(X_PATH).astype(np.float32)   # (N, Dx)
meta = pd.read_csv(META_PATH).reset_index(drop=True)

assert X.shape[0] == len(meta), "❌ 제품 벡터 수와 메타 행 수 불일치"

Dp = P.shape[1]
Dx = X.shape[1]

print(f"persona dim = {Dp}, product dim = {Dx}")

# -----------------------------------------------------------------------------
# 2) DIM ALIGN (자동 보정)
# -----------------------------------------------------------------------------
if Dp != Dx:
    if Dp > Dx:
        raise ValueError("❌ persona 벡터가 product보다 큼 (설계 오류)")

    pad_dim = Dx - Dp
    pad_vec = X[:, :pad_dim].mean(axis=0)

    P_fixed = []
    for v in P:
        P_fixed.append(np.concatenate([pad_vec, v]))

    P = np.stack(P_fixed).astype(np.float32)
    print(f"✅ persona vectors auto-aligned → {P.shape[1]} dim")

# -----------------------------------------------------------------------------
# 3) NORMALIZE
# -----------------------------------------------------------------------------
P_norm = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

# -----------------------------------------------------------------------------
# 4) SIMILARITY
# -----------------------------------------------------------------------------
S = P_norm @ X_norm.T   # (P, N)

persona_ids = [f"persona_{i}" for i in range(1, P.shape[0] + 1)]

brand_col = (
    "brand" if "brand" in meta.columns
    else "브랜드" if "브랜드" in meta.columns
    else None
)

# -----------------------------------------------------------------------------
# 5) FULL SAVE
# -----------------------------------------------------------------------------
full_rows = []
for pi, pid in enumerate(persona_ids):
    sims = S[pi]
    for j, sim in enumerate(sims):
        row = {
            "persona_id": pid,
            "product_index": j,
            "similarity": float(sim),
        }
        if brand_col:
            row["brand"] = meta.loc[j, brand_col]
        full_rows.append(row)

df_full = pd.DataFrame(full_rows)
df_full.to_csv(OUT_FULL, index=False)
print("saved:", OUT_FULL, "rows:", len(df_full))

# -----------------------------------------------------------------------------
# 6) TOP-N SAVE
# -----------------------------------------------------------------------------
top_rows = []
for pi, pid in enumerate(persona_ids):
    sims = S[pi]
    top_idx = np.argsort(-sims)[:TOPN]
    for rank, j in enumerate(top_idx, start=1):
        row = {
            "persona_id": pid,
            "rank": rank,
            "product_index": j,
            "similarity": float(sims[j]),
        }
        if brand_col:
            row["brand"] = meta.loc[j, brand_col]
        top_rows.append(row)

df_top = pd.DataFrame(top_rows)
df_top.to_csv(OUT_TOPN, index=False)
print("saved:", OUT_TOPN, "rows:", len(df_top))

# -----------------------------------------------------------------------------
# 7) persona × brand × brand_tone_cluster
# -----------------------------------------------------------------------------
seg = pd.read_csv(SEG_PATH)

if "brand" not in seg.columns and "브랜드" in seg.columns:
    seg = seg.rename(columns={"브랜드": "brand"})
if "brand" not in df_full.columns and "브랜드" in df_full.columns:
    df_full = df_full.rename(columns={"브랜드": "brand"})

required = {"brand", "brand_tone_cluster"}
missing = required - set(seg.columns)
if missing:
    raise ValueError(f"❌ final_brand_segments.csv missing columns: {missing}")

merged = df_full.merge(
    seg[["brand", "brand_tone_cluster"]],
    on="brand",
    how="left"
)

persona_brand = (
    merged
    .groupby(["persona_id", "brand", "brand_tone_cluster"], dropna=False)
    .agg(
        avg_similarity=("similarity", "mean"),
        product_cnt=("product_index", "count")
    )
    .reset_index()
)

persona_brand.to_csv(OUT_STR, index=False)
print("saved:", OUT_STR, "rows:", len(persona_brand))