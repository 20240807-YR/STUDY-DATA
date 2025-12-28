# agent10/strategy_narrator.py
# ============================================================
# Strategy Narrator
# - ReAct Reasoning 결과를 바탕으로
# - 전략 설명 로그 + CRM 메시지 생성
# - 데이터 로딩 ❌
# - 계산 ❌ / 추천 ❌
# ============================================================

class StrategyNarrator:
    """
    [ Strategy Narrator ]
    - 전략 설명 (log)
    - 고객 발송 메시지 (message)
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    def generate(self, crm_row, reasoning, market_context=None):
        """
        Returns:
            message (str): 고객 발송 메시지
            log (str): 전략 설명 로그
        """

        # --------------------------------------------------
        # 1. 전략 설명 로그 (Explain)
        # --------------------------------------------------
        log_prompt = f"""
You are explaining a CRM strategy decision.

[CRM Row]
{crm_row}

[Reasoning]
{reasoning}

[Market Context]
{market_context}

Explain WHY this CRM strategy makes sense.
- Do NOT calculate
- Do NOT recommend alternatives
"""

        log = self.llm.explain(log_prompt)

        # --------------------------------------------------
        # 2. 고객 메시지 생성 (Generate)
        # --------------------------------------------------
        message_prompt = f"""
Based on the following CRM strategy explanation,
write a customer-facing CRM message.

[Strategy Explanation]
{log}

[CRM Row]
{crm_row}

Rules:
- Follow the tone implied by brand_tone_cluster
- Do NOT invent strategy
"""

        message = self.llm.generate(message_prompt)

        return message, log