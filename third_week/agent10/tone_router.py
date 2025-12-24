import numpy as np
from agent10.utils import cosine_similarity


def route_tone_vector(
    tone_vector: np.ndarray,
    tone_centroids: dict,
    tone_profile_map: dict
):
    """
    Parameters
    ----------
    tone_vector : np.ndarray
        shape (D,) - 입력 톤 벡터
    tone_centroids : dict
        {tone_id: np.ndarray (D,)}
    tone_profile_map : dict
        {tone_id: parameter dict}

    Returns
    -------
    dominant_tone_id : str
    tone_parameters : dict
    """

    similarities = {
        tone_id: cosine_similarity(tone_vector, centroid)
        for tone_id, centroid in tone_centroids.items()
    }

    dominant_tone = max(similarities, key=similarities.get)

    return dominant_tone, tone_profile_map[dominant_tone]