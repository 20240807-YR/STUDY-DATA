from typing import Any, Optional


class CRMExecutor:
    """
    Executor
    """

    def __init__(self, crm_loader, reasoning_agent, narrator, market_tool=None):
        self.crm_loader = crm_loader
        self.reasoning_agent = reasoning_agent
        self.narrator = narrator
        self.market_tool = market_tool

    def load_crm_results(self, persona_id: str, topk: int):
        return self.crm_loader.load_crm_results(persona_id, topk)

    def reason(self, crm_row: Any):
        return self.reasoning_agent.run(crm_row)

    def market_context(self, crm_row: Any):
        if self.market_tool:
            return self.market_tool.run(crm_row)
        return None

    def narrate(
        self,
        crm_row: Any,
        reasoning: Any,
        market_context: Optional[Any] = None,
    ):
        return self.narrator.generate(
            crm_row=crm_row,
            reasoning=reasoning,
            market_context=market_context,
        )


Executor = CRMExecutor
__all__ = ["CRMExecutor", "Executor"]