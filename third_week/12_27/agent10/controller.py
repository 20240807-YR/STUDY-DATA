# agent10/controller.py
from pathlib import Path

# ============================================================
# 내부 모듈 import (패키지 기준, 절대경로)
# ============================================================

from agent10.crm_loader import CRMResultLoader
from agent10.react_reasoning_agent import ReActReasoningAgent
from agent10.strategy_narrator import StrategyNarrator
from agent10.executor import CRMExecutor
from agent10.market_context_tool import MarketContextTool
from agent10.openai_client import OpenAIChatCompletionClient
from agent10.tone_profiles import load_tone_dict


# ============================================================
# Plan-and-Execute Controller
# ============================================================

class PlanAndExecuteController:
    def __init__(self, executor: CRMExecutor):
        self.executor = executor

    def run(self, persona_id, topk=3, use_market_context=False):
        crm_rows = self.executor.load_crm_results(persona_id, topk)

        results = []
        for row in crm_rows:
            # 1) ReAct Reasoning
            reasoning = self.executor.reason(row)

            # 2) Market Context (optional)
            market = None
            if use_market_context:
                market = self.executor.market_context(row)

            # 3) Strategy Narration
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
                "log": log,
            })

        return results


# ============================================================
# 🔑 main() — "조립 책임"은 무조건 여기서만
# ============================================================

def main(persona_id="persona_1", topk=3, use_market_context=False):
    """
    Agent10 전체 파이프라인 엔트리포인트
    (Plan-and-Execute 구조의 'Plan' 담당)
    """

    # 📁 경로 기준: agent10 폴더의 상위 = 12_27
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"

    # --------------------------------------------------------
    # 1) CRM Result Loader
    # --------------------------------------------------------
    crm_loader = CRMResultLoader(
        csv_path=str(DATA_DIR / "persona_brand_tone_part_final.csv")
    )

    # --------------------------------------------------------
    # 2) LLM Client (ChatCompletion only, 생성 주체 아님)
    # --------------------------------------------------------
    llm_client = OpenAIChatCompletionClient()

    # --------------------------------------------------------
    # 3) Tone Dictionary
    # --------------------------------------------------------
    tone_dict = load_tone_dict(DATA_DIR)

    # --------------------------------------------------------
    # 4) ReAct Reasoning Agent
    # --------------------------------------------------------
    reasoning_agent = ReActReasoningAgent(
        llm_client=llm_client,
        tone_dict=tone_dict
    )

    # --------------------------------------------------------
    # 5) Strategy Narrator
    # --------------------------------------------------------
    narrator = StrategyNarrator(llm_client=llm_client)

    # --------------------------------------------------------
    # 6) Market Context Tool (optional)
    # --------------------------------------------------------
    market_tool = MarketContextTool() if use_market_context else None

    # --------------------------------------------------------
    # 7) Executor (실행 책임)
    # --------------------------------------------------------
    executor = CRMExecutor(
        crm_loader=crm_loader,
        reasoning_agent=reasoning_agent,
        narrator=narrator,
        market_tool=market_tool
    )

    # --------------------------------------------------------
    # 8) Controller (Plan-and-Execute)
    # --------------------------------------------------------
    controller = PlanAndExecuteController(executor)

    return controller.run(
        persona_id=persona_id,
        topk=topk,
        use_market_context=use_market_context
    )