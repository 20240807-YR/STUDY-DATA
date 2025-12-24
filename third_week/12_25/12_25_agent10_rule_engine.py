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

# 3. Import
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
    # Analyzer 로드 (이때 pkl 파일도 Gemini 모델로 만들어져 있어야 차원이 맞습니다)
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

    # 5. 스키마 기준 정렬 (중복 제거 및 스키마 검증)
    validated_order = []
    for slot in order:
        if slot in slot_schema and slot not in validated_order:
            validated_order.append(slot)
            
    return validated_order


def render_slot_sentence(slot, params, product, persona):
    """
    개별 슬롯을 문장으로 렌더링하고, 금칙어/대체어를 적용합니다.
    """
    base_sentence = SLOT_TEMPLATES.get(slot, "")
    sentence = base_sentence.format(
        persona=persona,
        product=product,
        proof_point="임상 데이터",      # TODO: 실제 데이터 연동 필요
        emotion_phrase="편안한 일상"    # TODO: 실제 데이터 연동 필요
    )

    # Lexicon(어휘) 적용
    lex_group = params.get("lexicon_group")
    if lex_group in LEXICON:
        sentence += f" ({LEXICON[lex_group][0]})"

    # Ban Group(금칙어) 필터링
    ban_group = params.get("ban_group")
    if ban_group in BAN_WORDS:
        for w in BAN_WORDS[ban_group]:
            sentence = sentence.replace(w, "")

    return sentence.strip()


def generate_message(persona, tone_vector, slot_schema, product):
    """
    Agent의 메인 진입점
    """
    # 1. 톤 분석
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
# [5] 실행 테스트 (Google Gemini API 적용)
# ==============================================================================
if __name__ == "__main__":
    import google.generativeai as genai
    from dotenv import load_dotenv
    import pprint

    # .env 파일 로드
    load_dotenv()
    
    # 1. API 키 설정 (GOOGLE_API_KEY)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # 혹은 "GEMINI_API_KEY"
    
    if not GOOGLE_API_KEY:
        print("\n❌ Google API Key가 없습니다. .env 파일을 확인해주세요.")
    else:
        # Gemini 설정
        genai.configure(api_key=GOOGLE_API_KEY)
        
        print("\n🚀 Agent Rule Engine Test Started (with Google Gemini)...")
        
        # 2. 테스트할 입력 텍스트
        # (원하는 톤이 나오는지 확인하기 위해 명확한 문장을 입력해보세요)
        user_input_text = "장인정신이 깃든 최고의 걸작, 시간을 초월한 가치를 선사합니다." 
        # user_input_text = "임상 실험으로 검증된 99%의 주름 개선 효과를 데이터를 통해 확인하세요."
        
        print(f"📄 입력 문장: {user_input_text}")

        try:
            # 3. 실제 Gemini 임베딩 생성
            # models/text-embedding-004 모델 사용 (768차원)
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=user_input_text,
                task_type="retrieval_query"
            )
            
            real_vector = np.array(result['embedding'])
            print(f"✅ 임베딩 완료 (차원수: {len(real_vector)})")

            # 4. 차원 일치 여부 확인 (중요)
            # pkl 파일과 현재 API 모델의 차원이 다르면 계산 불가
            # analyzer.centroids의 첫 번째 값의 차원을 확인
            if hasattr(analyzer, 'centroids') and len(analyzer.centroids) > 0:
                first_key = list(analyzer.centroids.keys())[0]
                pkl_dim = len(analyzer.centroids[first_key])
                if len(real_vector) != pkl_dim:
                    print(f"\n⚠️ [경고] 차원 불일치 발생!")
                    print(f" - PKL 파일 벡터 차원: {pkl_dim}")
                    print(f" - 현재 API 벡터 차원: {len(real_vector)}")
                    print(" -> PKL 파일을 다시 만들거나, 모델을 통일해야 합니다.")
                    sys.exit(1)

            # 5. 메시지 생성 호출
            test_persona = "민지"
            test_product = "레티놀 세럼"
            test_schema = ["intro", "proof", "emotion", "cta"]

            final_result = generate_message(test_persona, real_vector, test_schema, test_product)

            # 6. 결과 출력
            print("\n[분석 및 생성 결과]")
            pprint.pprint(final_result)

        except Exception as e:
            print(f"\n❌ API 호출 또는 실행 중 오류 발생: {e}")