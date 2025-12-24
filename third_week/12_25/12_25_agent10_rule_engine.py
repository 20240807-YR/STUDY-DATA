# tone vectors
def route_tone(tone_vector, tone_centroids):
    sims = {
        tone_id: cosine_similarity(
            tone_vector.reshape(1, -1),
            centroid.reshape(1, -1)
        )[0, 0]
        for tone_id, centroid in tone_centroids.items()
    }
    return max(sims, key=sims.get)

# map_tone_vector_to_params(tone_vector)
from map_tone_vector_to_params import map_tone_vector_to_params

# decide_slot_order(params, slot_schema)
def decide_slot_order(params, slot_schema):
    """
    slot_schema: ["intro", "proof", "emotion", "cta"] 같은 리스트
    """

    order = []

    if params["proof_level"] == "high":
        order.append("proof")

    if params["emotion_level"] == "high":
        order.append("emotion")

    # 기본 intro는 항상 앞
    order = ["intro"] + order

    # CTA는 강도에 따라
    if params["cta_strength"] != "low":
        order.append("cta")

    # 중복 제거 + schema 기준 정렬
    final_order = []
    for slot in slot_schema:
        if slot in order and slot not in final_order:
            final_order.append(slot)

    return final_order

# render_slot_sentence(slot, params, product, persona)
SLOT_TEMPLATES = {
    "intro": "{persona}님께 {product}를 소개합니다.",
    "proof": "{product}는 {proof_point}에서 검증되었습니다.",
    "emotion": "{product}와 함께 {emotion_phrase}을 느껴보세요.",
    "cta": "지금 바로 경험해보세요."
}

LEXICON = {
    "scientific_words": ["검증", "임상", "데이터"],
    "emotional_words": ["따뜻한", "편안한", "행복한"]
}

BAN_WORDS = {
    "clinical_terms": ["질병", "치료"]
}

def render_slot_sentence(slot, params, product, persona):
    sentence = SLOT_TEMPLATES[slot].format(
        persona=persona,
        product=product,
        proof_point="임상 데이터",
        emotion_phrase="편안한 일상"
    )

    # lexicon 적용
    lex_group = params.get("lexicon_group")
    if lex_group in LEXICON:
        sentence += f" ({LEXICON[lex_group][0]})"

    # ban_group 필터
    ban_group = params.get("ban_group")
    if ban_group in BAN_WORDS:
        for w in BAN_WORDS[ban_group]:
            sentence = sentence.replace(w, "")

    return sentence.strip()

# generate_message (Agent 핵심)
def generate_message(persona, tone_vector, slot_schema, product):
    params = map_tone_vector_to_params(tone_vector)

    slot_order = decide_slot_order(params, slot_schema)

    sentences = []
    trace = {
        "tone_id": params["tone_id"],
        "similarity": params["similarity"],
        "slot_order": slot_order
    }

    for slot in slot_order:
        s = render_slot_sentence(slot, params, product, persona)
        sentences.append(s)

    return {
        "tone_id": params["tone_id"],
        "params": params,
        "slots": slot_order,
        "message": " ".join(sentences),
        "trace": trace
    }

