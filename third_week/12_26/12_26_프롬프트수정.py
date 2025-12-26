import os
from openai import OpenAI

# ============================================================
# V2 CRM MESSAGE GENERATOR (AGENT10 FORMAT)
# - LLM은 계산/추천/전략 결정 안 함
# - 이미 선택된 CRM 파이프라인 결과를
#   "마케팅 메시지 형식"으로 번역만 수행
# ============================================================

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("❌ OPENAI_API_KEY 환경변수 없음")


PROMPT_TEMPLATE = """
[ROLE]
당신은 CRM 파이프라인 결과를
마케팅 메시지 문장으로 변환하는 역할만 수행합니다.

당신은 계산, 추천, 랭킹, 전략 결정을 하지 않습니다.
입력으로 주어진 결과를 바꾸지 않고 그대로 사용합니다.

[INPUT: CRM_PIPELINE_RESULT]
{crm_result}

[INPUT: BRAND_TONE]
{tone_context}

[CONSTRAINTS]
- 새로운 상품이나 브랜드를 추가하지 마세요.
- 숫자, 점수, 순위를 새로 만들지 마세요.
- 입력에 없는 판단을 하지 마세요.
- 아래 출력 형식을 반드시 지키세요.

[OUTPUT FORMAT]
제목: (40자 이내, 1줄)

내용:
(350자 이내의 자연스러운 CRM 마케팅 메시지)
"""


def build_prompt(crm_result: str, tone_context: str) -> str:
    return PROMPT_TEMPLATE.format(
        crm_result=crm_result,
        tone_context=tone_context
    )


def call_llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You generate CRM marketing messages only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


# ============================================================
# 실행 테스트
# ============================================================
if __name__ == "__main__":

    crm_pipeline_result = """
persona_id: persona_2
brand: 라네즈
selected_product: 워터뱅크 크림
brand_tone_cluster: 1
avg_similarity: 0.812
ingredient_affinity: 0.74
"""

    tone_context = """
Cluster 1 tone:
- 맑고 산뜻한
- 수분 중심
- 가볍고 깨끗한 인상
"""

    prompt = build_prompt(crm_pipeline_result, tone_context)
    result = call_llm(prompt)

    print("=" * 40)
    print("[CRM MESSAGE OUTPUT]")
    print("=" * 40)
    print(result)