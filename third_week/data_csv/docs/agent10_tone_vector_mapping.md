아래 그대로 MD 초안으로 써도 되고,
복붙해서 바로 agent10_tone_vector_mapping.md로 저장해도 되는 수준으로 정리했어.

⸻

agent10_tone_vector_mapping.md

목적

본 문서는 tone_vector를 Agent 10 메시지 생성 파라미터로 변환하는 규칙을 고정하기 위한 설계 문서이다.
	•	기존 Tone Profile Template은 정성적 규칙
	•	본 문서는 정량 벡터 → 생성 제어 파라미터 변환 로직을 정의한다
	•	LLM 연결 이후에도 메시지 톤 일관성을 유지하는 것이 목적이다

⸻

1. Tone Vector 정의

Tone vector는 톤 카테고리별 키워드/문장 embedding을 기반으로 생성된 톤 centroid 벡터이다.

예시 tone_id:
	•	Scientific
	•	Emotional
	•	Luxury
	•	Casual

형태:

tone_vector: np.array (d,)

이 벡터는 persona_vector에 병합하지 않고,
메시지 생성 직전 조절 다이얼(dial) 로 사용한다.

⸻

2. Tone Vector → 생성 파라미터 매핑 개념

tone_vector는 직접 문장을 만들지 않는다.
대신 아래 생성 제어 파라미터를 결정한다.

제어 대상 파라미터
	•	sentence_len
	•	proof_level
	•	emotion_level
	•	cta_strength
	•	lexicon_group
	•	ban_group

⸻

3. Tone ID 기준 기본 매핑 테이블 (Rule-based v1)

tone_id	sentence_len	proof_level	emotion_level	cta_strength	lexicon_group	ban_group
Scientific	short	high	low	mid	technical	hype
Emotional	mid	low	high	low	warm	cold_fact
Luxury	short	mid	low	mid	prestige	discount
Casual	very_short	low	mid	high	casual	formal

📌 이 테이블은 tone_vector centroid와의 cosine similarity 최대값 기준으로 선택된다.

⸻

4. Vector 기반 보정 로직 (확장 설계)

tone_vector는 단일 tone_id로 hard routing 될 수도 있고,
혼합 톤일 경우 연속값 보정에 사용될 수 있다.

예시

emotion_level_score = dot(tone_vector, emotional_centroid)
proof_level_score   = dot(tone_vector, scientific_centroid)

이를 통해:
	•	emotion_level: low / mid / high
	•	proof_level: low / mid / high

를 threshold 기반으로 결정한다.

📌 초기 버전에서는 Rule-based 고정
📌 이후 ML 단계에서 회귀/분류 모델로 대체 가능

⸻

5. generate_sentence() 내부 반영 규칙

sentence_len
	•	very_short → 1문장, 10~12자
	•	short → 1문장, 15자 내외
	•	mid → 12문장, 2540자

proof_level
	•	high → 성분 / 수치 / 테스트 / 검증어 최소 1개 필수
	•	mid → 기능 또는 근거 단서 1개
	•	low → 근거 요소 생략 가능

emotion_level
	•	high → 공감어, 감정 형용사 허용
	•	mid → 정서어 제한적 사용
	•	low → 감탄사, 감정어 금지

cta_strength
	•	high → 명확한 행동 유도 (지금 확인하세요)
	•	mid → 선택적 유도
	•	low → 제안형 마무리

⸻

6. Agent 10 내 위치

tone_vector 매핑은 다음 단계에서 사용된다:

route_tone()
 → load_tone_profile()
 → map_tone_vector_to_params()   ← 본 문서
 → generate_sentence()

tone_vector는 생성 단계에서만 참조되며,
추천/랭킹/클러스터링에는 직접 사용하지 않는다.

⸻

7. 설계 원칙 요약
	•	tone_vector ≠ persona_vector
	•	tone_vector는 문체 제어용
	•	persona_vector는 대상 선택용
	•	생성과 선택을 분리함으로써 설명 가능성과 안정성 확보

