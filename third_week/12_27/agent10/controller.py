# agent10/controller.py
from pathlib import Path

from crm_loader import CRMResultLoader
from react_reasoning_agent import ReActReasoningAgent
from strategy_narrator import StrategyNarrator
from executor import CRMExecutor
from market_context_tool import MarketContextTool
from openai_client import OpenAIChatCompletionClient
from tone_profiles import load_tone_dict


class PlanAndExecuteController:
    def __init__(self, executor: CRMExecutor):
        self.executor = executor

    def run(self, persona_id, topk=3, use_market_context=False):
        crm_rows = self.executor.load_crm_results(persona_id, topk)

        results = []
        for row in crm_rows:
            reasoning = self.executor.reason(row)
            market = self.executor.market_context(row) if use_market_context else None

            message, log = self.executor.narrate(
                crm_row=row,
                reasoning=reasoning,
                market_context=market,
            )

            results.append({
                "crm_row": row,
                "reasoning": reasoning,
                "market_context": market,
                "message": message,
                "log": log,
            })

        return results


def main(persona_id="persona_1", topk=3, use_market_context=False):
    BASE_DIR = Path(__file__).resolve().parent          # agent10/
    DATA_DIR = BASE_DIR.parent / "data"                 # 12_27/data

    # 1) CRM Loader
    crm_loader = CRMResultLoader(
        csv_path=str(DATA_DIR / "persona_brand_tone_part_final.csv")
    )

    # 2) LLM Client
    # ✅ mock 파라미터 없음 (OPENAI_OFFLINE=1로만 제어)
    llm_client = OpenAIChatCompletionClient()

    # 3) Tone Dict
    tone_dict = load_tone_dict(DATA_DIR)

    # 4) ReAct Agent
    reasoning_agent = ReActReasoningAgent(
        llm_client=llm_client,
        tone_dict=tone_dict,
    )

    # 5) Narrator
    narrator = StrategyNarrator(llm_client=llm_client)

    # 6) Market Tool (optional)
    market_tool = MarketContextTool() if use_market_context else None

    # 7) Executor
    executor = CRMExecutor(
        crm_loader=crm_loader,
        reasoning_agent=reasoning_agent,
        narrator=narrator,
        market_tool=market_tool,
    )

    controller = PlanAndExecuteController(executor)
    return controller.run(persona_id, topk, use_market_context)