# agent10/crm_loader.py
import pandas as pd


class CRMResultLoader:
    """
    CRM 결과 CSV 로더
    - persona × brand × tone × score 결과를 로드
    - '추천/계산'이 아니라 이미 계산된 결과를 읽기만 함
    """

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

        # 최소 컬럼 체크 (제출용 안정성)
        required_cols = {"persona_id", "score"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"CRM CSV 필수 컬럼 누락: {missing}")

    def load_crm_results(self, persona_id: str, topk: int):
        """
        Executor가 호출하는 정식 인터페이스
        """
        sub = self.df[self.df["persona_id"] == persona_id]

        return (
            sub.sort_values("score", ascending=False)
               .head(topk)
               .to_dict("records")
        )