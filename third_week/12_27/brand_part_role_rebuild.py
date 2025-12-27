# ============================================================
# Brand Analysis Part Role Rebuild (KOR 컬럼 대응)
# - 입력: data/brand_analysis_result.csv
# - 출력: data/brand_analysis_part_enhanced.csv
# - brand / part_id / content / char_len 없으면 자동 매핑 + char_len 생성
# ============================================================

import pandas as pd
from pathlib import Path

# ----------------------------
# 경로 (12_27 기준 고정)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

IN_PATH  = DATA_DIR / "brand_analysis_result.csv"
OUT_PATH = DATA_DIR / "brand_analysis_part_enhanced.csv"

# ----------------------------
# 로드
# ----------------------------
if not IN_PATH.exists():
    raise FileNotFoundError(f"입력 CSV 없음: {IN_PATH}")

df = pd.read_csv(IN_PATH)

print("[INFO] INPUT:", IN_PATH)
print("[INFO] RAW COLS:", list(df.columns))
print("[INFO] ROWS:", len(df))

# ----------------------------
# 컬럼 매핑 (네가 준 컬럼 기준)
# 브랜드,구분,내용 요약,핵심 키워드,AI 분석 톤,확신도,형용사 수,embedding_vector
# ----------------------------
rename_map = {}

if "brand" not in df.columns and "브랜드" in df.columns:
    rename_map["브랜드"] = "brand"

if "part_id" not in df.columns and "구분" in df.columns:
    rename_map["구분"] = "part_id"

if "content" not in df.columns and "내용 요약" in df.columns:
    rename_map["내용 요약"] = "content"

if rename_map:
    df = df.rename(columns=rename_map)

# ----------------------------
# 필수 컬럼 보정
# ----------------------------
required = {"brand", "part_id", "content"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"필수 컬럼 누락: {missing} | 현재 컬럼: {list(df.columns)}")

# char_len 없으면 생성
if "char_len" not in df.columns:
    df["char_len"] = df["content"].astype(str).map(len)

# 정리
df["brand"] = df["brand"].astype(str).str.strip()
df["part_id"] = df["part_id"].astype(str).str.strip()

# ----------------------------
# part_role 분류 로직
# ----------------------------
def infer_part_role(text: str) -> str:
    t = str(text).lower()

    score = {
        "identity": 0,  # 브랜드 정체성/철학/헤리티지
        "proof": 0,     # 임상/기술/성분/연구
        "benefit": 0,   # 효능/결과/기능
        "emotion": 0,   # 감성/라이프스타일/세계관
        "ritual": 0,    # 사용 경험/루틴/과정
        "cta": 0,       # 행동 유도
    }

    identity_kw = [
        "brand", "identity", "heritage", "since", "journey",
        "origin", "about", "philosophy", "story", "apex", "contemporary",
        "철학", "헤리티지", "여정", "오리진", "브랜드", "스토리"
    ]
    proof_kw = [
        "clinical", "lab", "technology", "data", "science", "patent",
        "임상", "연구", "기술", "성분", "데이터", "검증", "특허", "테크"
    ]
    benefit_kw = [
        "effect", "result", "solution", "benefit", "performance",
        "효과", "개선", "솔루션", "보습", "장벽", "탄력", "윤기", "케어"
    ]
    emotion_kw = [
        "sensual", "holistic", "balance", "ritual", "nature",
        "감성", "일상", "행복", "자연", "숲", "향", "균형", "힐링"
    ]
    ritual_kw = [
        "use", "routine", "step", "layering", "experience",
        "사용", "루틴", "단계", "과정", "사용감", "사용법"
    ]
    cta_kw = [
        "discover", "try", "start", "join", "now", "shop",
        "지금", "선택", "경험", "함께", "구매", "만나", "바로", "추천"
    ]

    for k in identity_kw:
        if k in t: score["identity"] += 1
    for k in proof_kw:
        if k in t: score["proof"] += 1
    for k in benefit_kw:
        if k in t: score["benefit"] += 1
    for k in emotion_kw:
        if k in t: score["emotion"] += 1
    for k in ritual_kw:
        if k in t: score["ritual"] += 1
    for k in cta_kw:
        if k in t: score["cta"] += 1

    # 아무것도 안 걸리면 identity로
    if max(score.values()) == 0:
        return "identity"

    return max(score, key=score.get)

# ----------------------------
# 적용
# ----------------------------
df["part_role"] = df["content"].apply(infer_part_role)

# ----------------------------
# 저장
# ----------------------------
df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("=" * 60)
print("brand_analysis_part_enhanced.csv 생성 완료")
print("ROWS:", len(df))
print("OUT:", OUT_PATH)
print("COLS:", list(df.columns))
print("=" * 60)