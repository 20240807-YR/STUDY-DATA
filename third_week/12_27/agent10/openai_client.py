# openai_client.py
# ============================================================
# OpenAI ChatCompletion Client
# - 설명 / 전략 해석 전용
# - 계산 ❌ / 추천 ❌
# ============================================================

import os
from openai import OpenAI


class OpenAIChatCompletionClient:
    """
    OpenAI ChatCompletion Wrapper

    원칙:
    - 계산 ❌
    - 추천 ❌
    - 점수 생성 ❌
    - 오직 '설명'과 '전략적 문장화'만 수행
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY 환경변수가 설정되지 않았습니다."
            )

        self.client = OpenAI(api_key=api_key)
        self.model = model

    # --------------------------------------------------------
    # ReAct Reasoning Agent 전용
    # --------------------------------------------------------
    def explain(self, prompt: str) -> str:
        """
        전략 설명 전용 메서드

        - CoT(Chain of Thought)는 내부적으로만 사용
        - 최종 출력은 '설명 텍스트'만 반환
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a CRM strategy analyst.\n"
                            "Do NOT perform calculations.\n"
                            "Do NOT make recommendations.\n"
                            "Only explain the given CRM decision, "
                            "persona–brand fit, and tone strategy."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            raise RuntimeError(f"[OpenAI explain() error] {str(e)}")

    # --------------------------------------------------------
    # Strategy Narrator 전용
    # --------------------------------------------------------
    def generate(self, prompt: str) -> str:
        """
        CRM 메시지 생성 전용 메서드

        - Executor / Narrator에서만 호출
        - 전략은 이미 주어진 상태에서 문장화만 수행
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a senior marketing copywriter AI.\n"
                            "Follow the given strategy strictly.\n"
                            "Do NOT invent strategy or scores.\n"
                            "Generate CRM messages only."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            raise RuntimeError(f"[OpenAI generate() error] {str(e)}")