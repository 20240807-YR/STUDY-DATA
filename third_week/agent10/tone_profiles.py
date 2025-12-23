# tone_id → generation parameter profile

TONE_PROFILE_MAP = {
    "Scientific": {
        "sentence_len": "short",
        "proof_level": "high",
        "emotion_level": "low",
        "cta_strength": "mid",
        "lexicon_hint": ["성분", "효능", "검증", "임상"],
        "ban_hint": ["과장", "감탄", "유행어"]
    },

    "Emotional": {
        "sentence_len": "mid",
        "proof_level": "low",
        "emotion_level": "high",
        "cta_strength": "low",
        "lexicon_hint": ["위로", "일상", "함께"],
        "ban_hint": ["전문용어", "수치나열"]
    },

    "Luxury": {
        "sentence_len": "mid",
        "proof_level": "mid",
        "emotion_level": "mid",
        "cta_strength": "mid",
        "lexicon_hint": ["가치", "품격", "신뢰", "헤리티지"],
        "ban_hint": ["할인", "가성비", "유행어"]
    },

    "Casual": {
        "sentence_len": "very_short",
        "proof_level": "low",
        "emotion_level": "high",
        "cta_strength": "high",
        "lexicon_hint": ["지금", "간편", "추천"],
        "ban_hint": ["장황", "근거장문"]
    }
}