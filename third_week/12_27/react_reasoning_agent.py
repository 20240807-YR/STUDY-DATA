class ReActReasoningAgent:
    def __init__(self, llm_client, tone_dict):
        self.llm = llm_client
        self.tone_dict = tone_dict

    def run(self, crm_row):
        prompt = f"""
        You are a CRM strategy analyst.

        Self-Ask:
        1. Why does persona '{crm_row['persona_id']}' match brand '{crm_row['brand']}'?
        2. What does tone cluster '{crm_row['brand_tone_cluster']}' imply?
        3. How should this be explained as a CRM strategy?

        Tone dictionary:
        {self.tone_dict}
        """

        # ChatCompletion 호출 (CoT는 노출 안 됨)
        return self.llm.explain(prompt)