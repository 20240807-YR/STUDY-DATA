import pandas as pd
from pathlib import Path


def load_tone_dict(data_dir):
    """
    CSV 기반 톤 사전 로더
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / "tone_centroid_profile.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"tone_centroid_profile.csv 없음: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {"brand_tone_cluster", "tones"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"tone_profile CSV 컬럼 누락: {missing}")

    tone_dict = {}
    for _, row in df.iterrows():
        tone_dict[int(row["brand_tone_cluster"])] = str(row["tones"]).strip()

    return tone_dict