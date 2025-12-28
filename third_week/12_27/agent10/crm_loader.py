import pandas as pd


class CRMResultLoader:
    """
    CRM 결과 CSV 로더
    """

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

        required_cols = {"persona_id", "score"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"CRM CSV 필수 컬럼 누락: {missing}")

    def load_crm_results(self, persona_id: str, topk: int):
        sub = self.df[self.df["persona_id"] == persona_id]
        return (
            sub.sort_values("score", ascending=False)
               .head(topk)
               .to_dict("records")
        )