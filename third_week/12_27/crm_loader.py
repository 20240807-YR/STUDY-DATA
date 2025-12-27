class CRMResultLoader:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def load(self, persona_id, topk):
        sub = self.df[self.df["persona_id"] == persona_id]
        return sub.sort_values("score", ascending=False).head(topk).to_dict("records")