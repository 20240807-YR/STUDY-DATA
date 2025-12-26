import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

# -----------------------------------------------------------
# 1. 환경변수 로드 (보안 설정)
# -----------------------------------------------------------
# 현재 파일이 있는 위치를 기준으로 .env 파일을 찾습니다.
current_dir = Path(__file__).parent
env_path = current_dir / '.env'

# .env 파일 로드
load_dotenv(dotenv_path=env_path)

# 환경변수에서 API 키 가져오기
api_key = os.getenv("GEMINI_API_KEY")

# 키가 없는 경우 안전하게 에러 처리
if not api_key:
    print("❌ [오류] .env 파일을 찾을 수 없거나 GEMINI_API_KEY가 설정되지 않았습니다.")
    print("   .env 파일에 'GEMINI_API_KEY=AIza...' 형태로 저장되어 있는지 확인해주세요.")
    exit()

def generate_marketing_agent_response(crm_input: str, tone_context: str):
    """
    requests 라이브러리를 사용하여 Gemini API를 호출하는 함수 (인코딩 문제 해결됨)
    """
    
    # 성공했던 그 모델 이름
    model_name = "gemini-flash-latest"
    
    # API 엔드포인트 URL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # 프롬프트 내용 구성
    payload = {
        "contents": [{
            "parts": [{
                "text": f"""
                [Role]
                당신은 아모레퍼시픽의 수석 마케팅 카피라이터 AI입니다.
                
                [Tone & Manner]
                {tone_context}
                
                [CRM Data]
                {crm_input}
                
                [Instruction]
                위 데이터를 분석하여 마케팅 메시지를 작성하세요.
                다음 순서로 사고(Chain of Thought)하여 작성하세요:
                1. 분석(Analyze) -> 2. 전략(Strategy) -> 3. 생성(Action)
                """
            }]
        }],
        "generationConfig": {
            "temperature": 0.7
        }
    }

    try:
        print(f"🚀 [{model_name}] 모델에게 요청을 보냅니다... (API Key 보안 적용됨)")
        
        # 윈도우 한글 깨짐 방지를 위한 인코딩 처리
        response = requests.post(
            url, 
            headers=headers, 
            data=json.dumps(payload, ensure_ascii=False).encode('utf-8')
        )
        
        # 결과 처리
        if response.status_code == 200:
            result_json = response.json()
            try:
                # 응답 텍스트 추출
                text = result_json['candidates'][0]['content']['parts'][0]['text']
                return text
            except KeyError:
                return f"⚠️ 응답은 왔지만 내용이 비어있습니다.\n전체 응답: {result_json}"
        else:
            return f"❌ 요청 실패 (코드 {response.status_code}): {response.text}"

    except Exception as e:
        return f"❌ 연결 오류: {str(e)}"

# ==========================================
# 실행부
# ==========================================
if __name__ == "__main__":
    sample_crm = "고객명: 김지민, 피부타입: 수부지, 최근구매: 라네즈 워터뱅크"
    sample_tone = "- 친근하고 생기 있는 어조 (이모지 활용 💧)"

    result = generate_marketing_agent_response(sample_crm, sample_tone)
    
    print("\n" + "="*30)
    print("[최종 결과]")
    print("="*30)
    print(result)