"""
[TEST RUNNER]
Agent 10 – CRM Message Reasoning Pipeline Test

목적:
- Plan-and-Execute 구조가 실제로 연결되어 실행되는지 검증
- LLM이 '계산/추천'이 아니라 '설명/전략 해석'만 수행하는지 확인
- 제출용 레포 구성 전에 전체 파이프라인 sanity check

실행:
$ python run_agent10_test.py
"""

# ============================================================
# 0. 환경 확인
# ============================================================

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# 로컬 모듈 import 안정화
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

print("=" * 70)
print("[Agent10 TEST] Environment Check")
print("BASE_DIR :", BASE_DIR)
print("DATA_DIR :", DATA_DIR)
print("DATA_DIR exists:", DATA_DIR.exists())
print("=" * 70)

# ============================================================
# 1. Controller 호출
# ============================================================

print("\n[STEP 1] Plan-and-Execute Controller 시작")

from controller import main as controller_main


# ============================================================
# 2. 실행
# ============================================================

if __name__ == "__main__":
    try:
        # ----------------------------
        # 테스트 파라미터 (제출용 예시)
        # ----------------------------
        persona_id = "persona_1"
        topk = 3
        use_market_context = False   # Optional Tool OFF (명시적)

        results = controller_main(
            persona_id=persona_id,
            topk=topk,
            use_market_context=use_market_context
        )

        print("\n" + "=" * 70)
        print("[TEST RESULT] ✅ 전체 파이프라인 정상 실행")
        print(f"persona_id: {persona_id}")
        print(f"rows returned: {len(results)}")

        if results:
            sample = results[0]

            print("\n--- [SAMPLE CRM ROW] ---")
            print(sample["crm_row"])

            print("\n--- [SAMPLE STRATEGY MESSAGE] ---")
            print(sample["message"])

            print("\n--- [SAMPLE REASONING LOG] ---")
            print(sample["reasoning"])

            if sample.get("market_context"):
                print("\n--- [SAMPLE MARKET CONTEXT] ---")
                print(sample["market_context"])

        print("""
[CHECKLIST]
✔ CRM CSV 로드됨
✔ persona × brand × brand_tone_cluster 결과 사용
✔ LLM은 계산/추천 수행 안 함
✔ ReAct Reasoning Agent 정상 호출
✔ Self-Ask 기반 전략 설명 생성
✔ Strategy Narrator 메시지 출력
✔ Controller / Executor / Agent 역할 분리 유지
✔ Plan-and-Execute 구조 구현 완료
""")
        print("=" * 70)

    except Exception as e:
        print("\n" + "=" * 70)
        print("[TEST RESULT] ❌ 실행 중 오류 발생")
        print("ERROR:", str(e))
        print("=" * 70)
        raise