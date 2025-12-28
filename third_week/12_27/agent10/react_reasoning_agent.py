class ReActReasoningAgent:
    """
    설명 전용 ReAct Reasoning Agent
    """

    def __init__(self, llm_client, tone_dict):
        self.llm = llm_client
        self.tone_dict = tone_dict

    def run(self, crm_row: dict) -> str:
        prompt = f"""
[Context]
Persona: {crm_row.get("persona_id")}
Brand: {crm_row.get("brand")}
Tone Cluster: {crm_row.get("brand_tone_cluster")}
Part Role: {crm_row.get("part_role")}
Score: {crm_row.get("score")}

[Tone Dictionary]
{self.tone_dict}

Explain the CRM strategy.
Do NOT calculate or recommend.
"""
        return self.llm.explain(prompt)