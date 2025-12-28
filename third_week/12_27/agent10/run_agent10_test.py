# agent10/run_agent10_test.py
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent          # agent10/
ROOT_DIR = BASE_DIR.parent                          # 12_27/
DATA_DIR = ROOT_DIR / "data"                        # 12_27/data

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

print("=" * 70)
print("[Agent10 TEST] Environment Check")
print("BASE_DIR :", BASE_DIR)
print("ROOT_DIR :", ROOT_DIR)
print("DATA_DIR :", DATA_DIR)
print("DATA_DIR exists:", DATA_DIR.exists())
print("OPENAI_OFFLINE:", os.getenv("OPENAI_OFFLINE", "0"))
print("=" * 70)

from controller import main as controller_main

if __name__ == "__main__":
    try:
        results = controller_main(
            persona_id="persona_1",
            topk=3,
            use_market_context=False,
        )

        print("\n[TEST RESULT] 실행 완료")
        print(f"rows: {len(results)}")

        if results:
            print("\n[SAMPLE MESSAGE]")
            print(results[0]["message"])

            print("\n[SAMPLE REASONING]")
            print(results[0]["reasoning"])

            # MOCK 여부 즉시 판별
            if "[MOCK" in str(results[0]["message"]) or "[MOCK" in str(results[0]["reasoning"]):
                print("\n[DIAG] 현재 결과는 MOCK 응답입니다.")
                print(" - OPENAI_OFFLINE=1 이거나 OpenAI 호출이 불가능한 상태입니다.")
                print(" - 실제 호출을 원하면 OPENAI_OFFLINE=0 이어야 하고, OPENAI API quota/billing이 유효해야 합니다.")

    except Exception as e:
        print("\n[TEST RESULT] ❌ 실행 실패")
        print("ERROR:", str(e))
        raise