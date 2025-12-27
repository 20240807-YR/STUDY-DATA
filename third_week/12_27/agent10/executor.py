# agent10/executor.py
# ============================================================
# Agent10 Executor
# ============================================================

from __future__ import annotations
from typing import Any, Optional


class CRMExecutor:
    """
    [ Executor ]
      ├─ CRM Result Loader (CSV / DB)
      ├─ ReAct Reasoning Agent (ChatCompletion API)
      ├─ Market Context Tool (Optional)
      └─ Strategy Narrator
    """

    def __init__(self, crm_loader, reasoning_agent, narrator, market_tool=None):
        self.crm_loader = crm_loader
        self.reasoning_agent = reasoning_agent
        self.narrator = narrator
        self.market_tool = market_tool

    # ========================================================
    # CRM Result Loader
    # ========================================================
    def load_crm_results(self, persona_id: str, topk: int):
        return self.crm_loader.load_crm_results(persona_id, topk)

    # ========================================================
    # ReAct Reasoning Agent
    # ========================================================
    def reason(self, crm_row: Any):
        return self.reasoning_agent.run(crm_row)

    # ========================================================
    # Market Context Tool (Optional)
    # ========================================================
    def market_context(self, crm_row: Any):
        if self.market_tool:
            return self.market_tool.run(crm_row)
        return None

    # ========================================================
    # Strategy Narrator
    # ========================================================
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


# ------------------------------------------------------------
# Controller 호환 alias (정석)
# ------------------------------------------------------------
Executor = CRMExecutor

__all__ = ["CRMExecutor", "Executor"]