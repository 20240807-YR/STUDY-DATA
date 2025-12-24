import sys
import os
import numpy as np
from pathlib import Path

# ==============================================================================
# [1] 경로 설정 & 모듈 Import 처리
# ==============================================================================
# 현재 파일 위치: .../third_week/12_25/
current_file_path = Path(__file__).resolve()
current_dir = current_file_path.parent

# 1. 12_24 폴더(모듈 있는 곳)를 sys.path에 추가
module_dir = current_dir.parent / "12_24"
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

# 2. 데이터 파일(pkl, csv) 경로 설정
data_dir = current_dir.parent / "data_csv"
path_centroids = data_dir / "tone_vectors.pkl"
path_meta = data_dir / "tone_metadata_extended.csv"

# 3. Import (이제 경로가 추가되었으므로 에러 없이 불러옵니다)
try:
    from map_tone_vector_to_params import ToneAnalyzer
    print("✅ 모듈 Import 성공")
except ImportError as e:
    print(f"❌ 모듈 Import 실패: {e}")
    print(f"참조 경로: {module_dir}")
    sys.exit(1)

# ==============================================================================
# [2] Analyzer 초기화 (전역 객체 생성)
# ==============================================================================
try:
    # 여기서 파일을 한 번만 로드합니다.
    analyzer = ToneAnalyzer.from_files(path_centroids, path_meta)
    print("✅ ToneAnalyzer 데이터 로드 완료")
except FileNotFoundError as e:
    print(f"❌ 데이터 파일 없음: {e}")
    sys.exit(1)


# ==============================================================================
# [3] 룰 엔진 설정 (Templates, Lexicon)
# ==============================================================================
SLOT_TEMPLATES = {
    "intro": "{persona}님께 {product}를 소개합니다.",
    "proof": "{product}는 {proof_point}에서 검증되었습니다.",
    "emotion": "{product}와 함께 {emotion_phrase}을 느껴보세요.",
    "cta": "지금 바로 경험해보세요."
}

LEXICON = {
    "scientific_words": ["검증", "임상", "데이터"],
    "emotional_words": ["따뜻한", "편안한", "행복한"]
}

BAN_WORDS = {
    "clinical_terms": ["질병", "치료"]
}


# ==============================================================================
# [4] 핵심 로직 함수
# ==============================================================================

def decide_slot_order(params, slot_schema):
    """
    톤 파라미터(params)에 따라 슬롯 순서를 동적으로 조정합니다.
    """
    order = []

    # 1. Proof 강조
    if params.get("proof_level") == "high":
        order.append("proof")

    # 2. Emotion 강조
    if params.get("emotion_level") == "high":
        order.append("emotion")

    # 3. Intro는 항상 맨 앞
    order = ["intro"] + order

    # 4. CTA 강도에 따른 추가
    if params.get("cta_strength") != "low":
        order.append("cta")

    # 5. 스키마 기준 정렬 및 중복 제거
    final_order = []
    # slot_schema에 있는 순서대로 정렬하되, order에 포함된 것만 남김
    # (단, 위 로직은 동적 추가된 슬롯이 스키마에 없으면 무시되므로, 
    #  여기서는 간단히 '사용자가 정의한 스키마 순서'를 우선하되 활성화된 슬롯만 뽑습니다)
    
    # *수정 제안*: 위에서 만든 `order`가 동적으로 생성된 순서이므로 이것을 유지하는 것이 맞습니다.
    # 다만, slot_schema에 없는 엉뚱한 슬롯이 들어가는 걸 방지하려면 아래처럼 합니다.
    
    validated_order = []
    for slot in order:
        if slot in slot_schema and slot not in validated_order:
            validated_order.append(slot)
            
    return validated_order


def render_slot_sentence(slot, params, product, persona):
    """
    개별 슬롯을 문장으로 렌더링하고, 금칙어/대체어를 적용합니다.
    """
    # 템플릿 채우기 (데이터는 예시로 하드코딩 되어있으나, 실제론 DB 등에서 가져와야 함)
    base_sentence = SLOT_TEMPLATES.get(slot, "")
    sentence = base_sentence.format(
        persona=persona,
        product=product,
        proof_point="임상 데이터",      # TODO: 실제 제품 데이터 연동 필요
        emotion_phrase="편안한 일상"    # TODO: 실제 감성 데이터 연동 필요
    )

    # Lexicon(어휘) 적용
    lex_group = params.get("lexicon_group")
    if lex_group in LEXICON:
        # 간단히 괄호로 추가하는 로직 유지
        sentence += f" ({LEXICON[lex_group][0]})"

    # Ban Group(금칙어) 필터링
    ban_group = params.get("ban_group")
    if ban_group in BAN_WORDS:
        for w in BAN_WORDS[ban_group]:
            sentence = sentence.replace(w, "")

    return sentence.strip()


def generate_message(persona, tone_vector, slot_schema, product):
    """
    Agent의 메인 진입점입니다.
    벡터 -> 톤 분석 -> 슬롯 순서 결정 -> 문장 생성 -> 결과 반환
    """
    
    # 1. 톤 분석 (Class Instance 사용)
    # 기존 route_tone 함수는 Analyzer 내부에 로직이 포함되었으므로 삭제됨
    try:
        params = analyzer.map_tone_vector_to_params(tone_vector)
    except Exception as e:
        return {"error": f"Tone analysis failed: {e}"}

    # 2. 슬롯 순서 결정
    slot_order = decide_slot_order(params, slot_schema)

    # 3. 문장 생성
    sentences = []
    for slot in slot_order:
        s = render_slot_sentence(slot, params, product, persona)
        sentences.append(s)

    # 4. 결과 조립
    trace = {
        "tone_id": params["tone_id"],
        "similarity": params["similarity"],
        "original_params": params,
        "final_slot_order": slot_order
    }

    return {
        "status": "success",
        "tone_id": params["tone_id"],
        "message": " ".join(sentences),
        "trace": trace
    }


# ==============================================================================
# [5] 실행 테스트
# ==============================================================================
if __name__ == "__main__":
    print("\n🚀 Agent Rule Engine Test Started...")
    
    # 테스트용 더미 데이터
    test_persona = "민지"
    test_product = "레티놀 세럼"
    test_schema = ["intro", "proof", "emotion", "cta"]
    
    # 임의의 벡터 생성 (실제 환경에선 임베딩 모델 출력값)
    # 768차원 (googleAI embedding size 예시)
    dummy_vector = np.random.rand(768)

    # 메시지 생성 호출
    result = generate_message(test_persona, dummy_vector, test_schema, test_product)
    
    import pprint
    pprint.pprint(result)
