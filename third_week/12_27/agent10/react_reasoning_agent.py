# react_reasoning_agent.py
# ============================================================
# ReAct Reasoning Agent
# - 설명 / 전략 해석 전용
# - 계산 ❌, 추천 ❌
# - CoT는 내부 추론용, 출력에는 노출 안 됨
# ============================================================

from typing import Dict, Any


class ReActReasoningAgent:
    def __init__(self, llm_client, tone_dict: Dict[str, Any]):
        """
        llm_client : OpenAIChatCompletionClient
        tone_dict  : tone_profiles.load_tone_dict() 결과
        """
        self.llm = llm_client
        self.tone_dict = tone_dict

    def run(self, crm_row: Dict[str, Any]) -> str:
        """
        입력:
            crm_row (dict)
            - persona_id
            - brand
            - brand_tone_cluster
            - score
            - part_role (있으면 사용)

        출력:
            전략 설명 텍스트 (str)
        """

        persona_id = crm_row.get("persona_id", "")
        brand = crm_row.get("brand", "")
        tone_cluster = crm_row.get("brand_tone_cluster", "")
        part_role = crm_row.get("part_role", "unknown")
        score = crm_row.get("score", None)

        prompt = f"""
You are a CRM strategy analyst.

[Context]
- Persona ID: {persona_id}
- Brand: {brand}
- Tone Cluster: {tone_cluster}
- Message Purpose (part_role): {part_role}
- Similarity Score: {score}

[Self-Ask]
1. Why does this persona match this brand?
2. What does this tone cluster imply in terms of brand communication?
3. How should this be explained as a CRM strategy (NOT a recommendation)?

[Tone Dictionary Reference]
{self.tone_dict}

[Instruction]
- Do NOT calculate anything.
- Do NOT recommend actions.
- Only explain the strategic reasoning behind this match.
"""

        # ChatCompletion 호출
        return self.llm.explain(prompt)