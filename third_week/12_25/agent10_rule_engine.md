1. 목적 (Why this exists)

Agent10은 ML 없이도
	•	톤 결정
	•	문장 구조 결정
	•	단어 선택
	•	결과 설명(trace)

이 모두가 결정론적(rule-based) 으로 이루어지는 Agent이다.

“왜 이 메시지가 나왔는지 설명 가능”
→ 이것이 Agent10의 핵심 목표다.

⸻

2. 전체 흐름 요약

tone_vector
 → route_tone
 → map_tone_vector_to_params
 → decide_slot_order
 → render_slot_sentence (slot별)
 → generate_message


⸻

3. Slot Order Decision Rules (A)

3.1 기본 원칙
	•	메시지는 항상 intro로 시작
	•	slot_schema가 최종 순서를 제한함 (허용된 slot만 사용)
	•	파라미터는 순서에만 영향, 문장 생성은 B에서 처리

⸻

3.2 proof_level 규칙

high
	•	신뢰·검증이 핵심 메시지
	•	proof slot을 intro 다음에 배치

이유:
	•	데이터/임상/효능 중심 브랜드 톤에서는
“왜 믿어야 하는가”가 가장 중요

mid / low
	•	proof를 강하게 밀지 않음
	•	intro → emotion / intro → CTA 흐름 유지

⸻

3.3 emotion_level 규칙

high
	•	감정·공감이 메시지 중심
	•	emotion slot을 앞쪽 또는 중간에 배치

이유:
	•	공감 → 설득 → 행동 유도 흐름

⸻

3.4 cta_strength 규칙

high / mid
	•	CTA는 반드시 포함
	•	메시지 마지막에 위치

low
	•	노골적인 행동 유도 제거
	•	정보 제공형 메시지 유지

⸻

3.5 최종 Slot Order 결정 방식
	1.	intro는 항상 포함
	2.	proof / emotion / cta 후보를 params로 결정
	3.	slot_schema 기준으로 허용된 slot만 남김
	4.	중복 제거 후 순서 확정

⸻

4. Slot Rendering Rules (B)

4.1 Slot Rendering의 역할

slot은 단순 문장이 아니라:
	•	템플릿
	•	단어 선택 규칙
	•	금지어 필터
	•	trace 기록

을 포함한 설명 가능한 생성 단위다.

⸻

4.2 Slot Template 구조

SLOT_TEMPLATES = {
    "intro": "{persona}님께 {product}를 소개합니다.",
    "proof": "{product}는 {proof_point}에서 검증되었습니다.",
    "emotion": "{product}와 함께 {emotion_phrase}을 느껴보세요.",
    "cta": "지금 바로 경험해보세요."
}


⸻

4.3 Lexicon Rule
	•	lexicon_group에 따라 사용 가능한 단어 집합 결정
	•	문장에 자연스럽게 삽입

예:
	•	scientific_words → 검증, 데이터, 임상
	•	emotional_words → 따뜻한, 편안한, 행복한

⸻

4.4 Ban Group Rule
	•	ban_group에 포함된 단어는 최종 문장에서 제거
	•	예:
	•	clinical_terms → 질병, 치료 제거

⸻

4.5 Trace 기록 (중요)

각 slot 생성 시 반드시 기록:

{
  "slot": "proof",
  "template": "...",
  "lexicon_group": "scientific_words",
  "used_words": ["검증"],
  "ban_group": "clinical_terms"
}

👉 이 trace가 있어야 Agent로 인정

⸻

5. generate_message 최종 출력

{
  "tone_id": "Scientific",
  "params": {...},
  "slots": ["intro", "proof", "cta"],
  "message": "...",
  "trace": {...}
}


⸻

6. 설계 철학 요약
	•	ML 없음
	•	확률 없음
	•	랜덤 없음
	•	모든 결과는 규칙과 파라미터로 설명 가능

이 구조가 완성되면 LLM 없이도 Agent다
