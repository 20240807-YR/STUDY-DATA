class StrategyNarrator:
    def __init__(self, llm_client):
        self.llm = llm_client

    def generate(self, crm_row, reasoning, market_context=None):
        prompt = f"""
        CRM Strategy Summary:
        - Persona: {crm_row['persona_id']}
        - Brand: {crm_row['brand']}
        - Purpose: {crm_row['part_role']}

        Reasoning:
        {reasoning}

        Market Context:
        {market_context}

        Generate:
        - Title (<=40 chars)
        - Body (<=350 chars)
        """

        text = self.llm.generate(prompt)

        return text, {
            "used_reasoning": True,
            "used_market_context": market_context is not None
        }