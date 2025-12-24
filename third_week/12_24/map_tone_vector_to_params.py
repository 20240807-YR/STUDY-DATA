import numpy as np
import pandas as pd
import pickle as pkl
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# ==================================================
# 경로 설정 (third_week 기준)
# ==================================================
BASE_DIR = Path(__file__).resolve().parents[2]   # STUDY-DATA/third_week
DATA_DIR = BASE_DIR / "data_csv"

PATH_CENTROIDS = DATA_DIR / "tone_vectors.pkl"
PATH_META = DATA_DIR / "tone_metadata_extended.csv"

# ==================================================
# 데이터 로드 (module import 시 1회)
# ==================================================
with open(PATH_CENTROIDS, "rb") as f:
    tone_centroids: dict[str, np.ndarray] = pkl.load(f)

tone_meta = pd.read_csv(PATH_META).set_index("tone_id")

# ==================================================
# 핵심 함수
# ==================================================
def map_tone_vector_to_params(tone_vector: np.ndarray) -> dict:
    """
    입력
    - tone_vector: np.ndarray (D,)

    출력
    - {
        tone_id,
        dominant_trait,
        proof_level,
        emotion_level,
        cta_strength,
        lexicon_group,
        ban_group,
        description,
        similarity
      }
    """

    sims = {}

    for tone_id, centroid in tone_centroids.items():
        sim = cosine_similarity(
            tone_vector.reshape(1, -1),
            centroid.reshape(1, -1)
        )[0, 0]
        sims[tone_id] = float(sim)

    # 가장 유사한 tone
    selected_tone = max(sims, key=sims.get)

    row = tone_meta.loc[selected_tone]

    return {
        "tone_id": selected_tone,
        "dominant_trait": row.get("dominant_trait"),
        "proof_level": row.get("proof_level"),
        "emotion_level": row.get("emotion_level"),
        "cta_strength": row.get("cta_strength"),
        "lexicon_group": row.get("lexicon_group"),
        "ban_group": row.get("ban_group"),
        "description": row.get("description"),
        "similarity": sims[selected_tone],
    }