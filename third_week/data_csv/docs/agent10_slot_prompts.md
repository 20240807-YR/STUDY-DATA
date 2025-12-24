목적

Tone Profile + Slot Schema + Rule Table을
실제 LLM 프롬프트로 연결하는 언어 설계 레이어


공통 전제 (모든 슬롯 공통)
	•	tone_profile_template.csv의 규칙을 절대 위반하지 말 것
	•	message_slot_schema.csv의 required_fields 미충족 시 생성 중단
	•	ban_hint 포함 시 즉시 재생성
	•	슬롯 1개 = sentence_len 규칙 내 최대 문장 수 고정



<< SLOT별 프롬프트 템플릿 >>

1️⃣ HOOK

목적
	•	첫 문장 주목도 확보
	•	톤 방향 즉시 인지

프롬프트

당신은 {brand_position} 톤의 메시지를 작성합니다.
목표는 {message_goal}입니다.

아래 조건을 만족하는 HOOK 문장을 생성하세요:
- sentence_len 규칙을 따를 것
- 감정 수준은 emotion_level에 맞출 것
- 과장/유행어/감탄사는 ban_hint에 따라 금지
- 브랜드 이름은 직접 언급하지 말 것


2️⃣ EMPATHY

목적
	•	페르소나 상태 공감
	•	관계 진입

프롬프트

{persona_skin_state} 또는 {season_context}를 기반으로
공감 문장을 생성하세요.

조건:
- 명령형 문장 금지
- 공포/압박 표현 금지
- emotion_level이 low인 경우 공감어 사용 금지


3️⃣ PROOF

목적
	•	신뢰 확보
	•	클리니컬 톤 핵심

프롬프트

아래 정보를 기반으로 근거 문장을 작성하세요:
- key_ingredient 또는 clinical_point 중 최소 1개 포함

조건:
- proof_level 규칙을 반드시 충족
- 수치/검증/메커니즘 중 1개 이상 포함
- 감성 표현 금지

📌 proof_level=high → 실패 시 전체 메시지 생성 중단


4️⃣ BENEFIT

목적
	•	사용자 관점 효익 전달

프롬프트

사용자가 체감할 수 있는 효익을 설명하세요.

조건:
- 기능 나열 금지
- 결과 중심 표현 사용
- emotion_level에 따라 정서어 조절


5️⃣ PRODUCT

목적
	•	제품 식별

프롬프트

다음 정보를 자연스럽게 연결하세요:
- brand
- product_name
- category (선택)

조건:
- 수식어 최소화
- 객관적 톤 유지


6️⃣ VALUE

목적
	•	프리미엄/신뢰 설득

프롬프트

제품의 가치 또는 품질 신호를 설명하세요.

조건:
- 할인/가성비 표현 금지
- 유행어 금지
- heritage / quality 키워드 우선 사용


7️⃣ STORY

목적
	•	브랜드 서사 전달

프롬프트

brand_story_key를 활용해 짧은 스토리 문장을 작성하세요.

조건:
- 과거/전통/철학 중 1개 축만 사용
- 장문 서사 금지


8️⃣ EVENT

목적
	•	시즌/한정 후킹

프롬프트

event_name 또는 period를 활용해
지금 행동해야 할 이유를 전달하세요.

조건:
- 긴 설명 금지
- urgency_word 1개 포함


9️⃣ CTA

목적
	•	행동 유도

프롬프트

message_goal에 맞는 CTA 문장을 생성하세요.

조건:
- cta_strength 규칙 준수
- 마지막 슬롯에서만 생성
- 명령 강도는 규칙 초과 금지


생성 실패 처리 규칙
	•	필수 필드 누락 → 즉시 중단
	•	ban_hint 위반 2회 → tone_profile 오류
	•	proof_level 미충족 → 재생성 1회 후 중단

