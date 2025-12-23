# Agent 10 Message Examples

본 문서는 Agent 10 메시지 생성 엔진의
tone_profile_id별 **문장 구조 예시(reference)**를 정의한다.

- 실제 생성 문장이 아님
- 품질 기준 및 방향성 고정 목적

## clinical_explain (고기능/클리니컬 · 신규)

slot_order: HOOK > PROOF > BENEFIT > PRODUCT > CTA

HOOK  
민감해진 피부에는 ‘느낌’보다 근거가 먼저입니다.

PROOF  
병풀 유래 성분과 진정 메커니즘이 반복 테스트로 확인되었습니다.

BENEFIT  
외부 자극 후 붉은기와 당김을 빠르게 완화하는 데 도움을 줍니다.

PRODUCT  
에스트라 더마 리페어 크림

CTA  
지금 성분과 효능을 직접 확인해보세요.

## gentle_relationship (수분 중심 · 재방문)

slot_order: EMPATHY > PRODUCT > BENEFIT > CTA

EMPATHY  
요즘처럼 건조한 날엔 피부도 쉽게 지치죠.

PRODUCT  
라네즈 워터 슬리핑 마스크는

BENEFIT  
하루의 끝에서 수분 밸런스를 천천히 회복하도록 도와줍니다.

CTA  
오늘 밤도 피부가 편안할 수 있게 챙겨보세요.

## trendy_hook (트렌디/컬러 · 시즌)

slot_order: HOOK > BENEFIT > PRODUCT > CTA

HOOK  
지금 가장 많이 찍히는 컬러.

BENEFIT  
가볍게 발라도 분위기 완성.

PRODUCT  
에스쁘아 뉴 꾸뛰르 립스틱

CTA  
이번 시즌 컬러, 바로 확인하세요.

※ 주의
- 이 문서의 문장은 정답이 아님
- tone_profile_template.csv + message_slot_schema.csv가 우선
- 출력 결과가 이 예시의 결을 벗어나면 품질 이슈로 판단