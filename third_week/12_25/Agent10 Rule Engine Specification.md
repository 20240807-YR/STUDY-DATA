📄 Agent10 Rule Engine Specification

1. 목적 (Goal)

Agent10은 LLM 없이, 사전에 정의된 rule과 tone centroid를 기반으로
설명 가능한 마케팅 메시지를 생성하는 rule-based agent이다.

본 Agent의 핵심 목표는 다음과 같다:
	•	입력 tone_vector에 대해 결정적으로 tone_id를 선택
	•	tone_id에 매핑된 언어 파라미터를 기반으로
	•	메시지 구조(slot order)와 문장을 생성
	•	생성 결과에 대해 trace를 통해 “왜 이 문장이 나왔는지” 설명 가능

⸻

2. 입력 / 출력 정의

입력
	•	persona : 타겟 사용자 요약 정보
	•	tone_vector : np.ndarray (D,)
	•	slot_schema : 메시지 슬롯 구조
예: ["intro", "proof", "emotion", "cta"]
	•	product_meta : 제품 정보 (이름, 핵심 효능 등)

출력

{
  "tone_id": "Scientific",
  "params": {...},
  "slots": ["intro", "proof", "cta"],
  "message": "...",
  "trace": {...}
}


⸻

3. Tone Routing Rule

route_tone
	•	cosine similarity 기준
	•	입력 tone_vector와 모든 centroid 비교
	•	가장 유사한 tone_id를 선택

tone_id = argmax(cosine_similarity(tone_vector, centroid))

이 단계는 **확률적 판단이 아닌 결정적(rule-based)**이다.

⸻

4. map_tone_vector_to_params
	•	tone_id에 매핑된 언어 파라미터를 반환
	•	CSV(tone_metadata_extended.csv) 기준
	•	포함 파라미터:
	•	dominant_trait
	•	proof_level
	•	emotion_level
	•	cta_strength
	•	lexicon_group
	•	ban_group
	•	description
	•	similarity

📌 이 함수는 import-only 모듈로 사용되며
Agent 내부에서 직접 CSV 경로를 신경 쓰지 않는다.

⸻

5. Slot Order Decision Rules

proof_level
	•	high
→ 신뢰/근거 중심 메시지
→ proof slot을 앞쪽에 배치

emotion_level
	•	high
→ 공감/감정 흐름 강화
→ emotion slot을 앞 또는 중간에 배치

cta_strength
	•	low
→ CTA 생략 또는 약화
	•	mid/high
→ CTA slot 포함

⸻

6. Slot Rendering & Trace

render_slot_sentence

각 slot은 다음 정보를 trace에 반드시 기록한다.

기록 항목
	•	사용된 템플릿
	•	적용된 lexicon_group
	•	제거된 ban_group
	•	실제 사용된 단어 (used_lexicon)

trace 구조 예시

"trace": {
  "tone_id": "Scientific",
  "similarity": 0.82,
  "slot_order": ["intro", "proof", "cta"],
  "slots": {
    "proof": {
      "template": "...",
      "lexicon_group": "scientific_words",
      "ban_group": "emotional_words",
      "used_lexicon": "검증"
    }
  }
}

📌 이 trace가 Agent 인정의 핵심 근거다.
→ “왜 이 문장이 나왔는지” 기계가 설명 가능.

⸻

7. Agent 인정 기준

Agent10은 다음 조건을 만족한다:
	•	LLM 호출 없음
	•	모든 판단이 rule / cosine / table 기반
	•	결과 재현 가능
	•	trace를 통해 의사결정 설명 가능

⸻

4️⃣ ipynb는 어디에 쓰냐?

✅ 새로 하나 파는 게 맞다

third_week/12_24/agent10_demo.ipynb

여기서 할 일:
	•	dummy tone_vector 생성
	•	persona / product_meta 입력
	•	generate_message() 호출
	•	message + trace 출력 확인

👉 코드 개발은 .py / 검증은 ipynb

