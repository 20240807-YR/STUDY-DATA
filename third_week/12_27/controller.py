class PlanAndExecuteController:
    def __init__(self, executor):
        self.executor = executor

    def run(self, persona_id, topk=3, use_market_context=False):
        crm_rows = self.executor.load_crm_results(persona_id, topk)

        results = []
        for row in crm_rows:
            reasoning = self.executor.reason(row)

            market = None
            if use_market_context:
                market = self.executor.market_context(row)

            message, log = self.executor.narrate(
                crm_row=row,
                reasoning=reasoning,
                market_context=market
            )

            results.append({
                "crm_row": row,
                "reasoning": reasoning,
                "market_context": market,
                "message": message,
                "log": log
            })

        return results