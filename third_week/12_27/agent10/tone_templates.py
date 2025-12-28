import pandas as pd
from pathlib import Path

def load_tone_templates(data_dir):
    path = Path(data_dir) / "tone_profile_template.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    return df.set_index("tone_profile_id").to_dict("index")