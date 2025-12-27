class CRMExecutor:
    def __init__(self, crm_loader, reasoning_agent, narrator, market_tool=None):
        self.crm_loader = crm_loader
        self.reasoning_agent = reasoning_agent
        self.narrator = narrator
        self.market_tool = market_tool

    def load_crm_results(self, persona_id, topk):
        return self.crm_loader.load(persona_id, topk)

    def reason(self, crm_row):
        return self.reasoning_agent.run(crm_row)

    def market_context(self, crm_row):
        if self.market_tool:
            return self.market_tool.run(crm_row)
        return None

    def narrate(self, crm_row, reasoning, market_context=None):
        return self.narrator.generate(
            crm_row=crm_row,
            reasoning=reasoning,
            market_context=market_context
        )