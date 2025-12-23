import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    a, b: shape (D,)
    """
    return float(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    )