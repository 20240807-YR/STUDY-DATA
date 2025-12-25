# schemas.py
from typing import TypedDict
import numpy as np

class CRMInput(TypedDict):
    persona_id: str
    brand: str
    brand_tone_cluster: int
    avg_similarity: float
    tone_vector: np.ndarray
    product_cnt: int