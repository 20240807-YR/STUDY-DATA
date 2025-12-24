# Agent 10 Architecture

본 문서는 Agent 10 메시지 생성 엔진의
전체 흐름과 함수 단위 역할을 정의한다.

- 실제 코드 아님
- 설계 고정용 명세

# 전체 플로우 요약 
[Input]
→ Tone Routing
→ Tone Profile Load
→ Slot Order Build
→ Slot Field Binding
→ Slot-wise Sentence Generation
→ Message Assembly
→ Output

## Main Entry

generate_message(input):
    tone_profile = route_tone(input)
    tone_rules   = load_tone_profile(tone_profile)
    slot_order   = build_slot_order(tone_rules)
    slot_data    = bind_slot_fields(slot_order, input)
    message      = render_slots(slot_order, slot_data, tone_rules)
    return message

    ## Tone Routing

route_tone(input):
    - tone_decision_table.csv 조회
    - brand_tone_cluster + message_goal 기준
    - tone_profile_id 반환

## Tone Profile Load

load_tone_profile(tone_profile_id):
    - tone_profile_template.csv 조회
    - 반환 항목:
        - slot_order
        - sentence_len
        - proof_level
        - emotion_level
        - cta_strength
        - lexicon_hint
        - ban_hint

## Slot Order Build

build_slot_order(tone_rules):
    - slot_order 문자열 파싱
    - 예: "HOOK>PROOF>BENEFIT>CTA"

## Slot Field Binding

bind_slot_fields(slot_order, input):
    for slot in slot_order:
        - message_slot_schema.csv 조회
        - required_fields 확인
        - 하나라도 없으면 생성 중단

## Slot Rendering

render_slots(slot_order, slot_data, tone_rules):
    for slot in slot_order:
        sentence = generate_sentence(slot, slot_data, tone_rules)
    join sentences

## Local Sentence Generator

generate_sentence(slot, data, tone_rules):
    - lexicon_hint 반영
    - ban_hint 회피
    - sentence_len / emotion_level / cta_strength 조절

## Design Principles

- Tone은 구조로 통제, 문장은 자유
- Slot 단위로 실패 가능
- 규칙은 CSV, 판단은 엔진, 언어는 LLM


ㄴ