# agent10/openai_client.py
import os

OPENAI_OFFLINE = os.getenv("OPENAI_OFFLINE", "0") == "1"


class OpenAIChatCompletionClient:
    """
    - ONLINE : 실제 OpenAI 호출
    - OFFLINE: MOCK 응답 (구조 테스트용)
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

        if not OPENAI_OFFLINE:
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY 환경변수가 없습니다.")

            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None

    # --------------------------------------------------
    # ReAct Reasoning
    # --------------------------------------------------
    def explain(self, prompt: str) -> str:
        if OPENAI_OFFLINE:
            return (
                "[MOCK EXPLANATION]\n"
                "This CRM strategy explains why the persona–brand fit is valid.\n"
                "The tone cluster supports consistent emotional framing.\n"
                "No calculation or recommendation is performed."
            )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a CRM strategy analyst. "
                            "Do NOT calculate or recommend. "
                            "Only explain the strategy."
                        )
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"[OpenAI explain() FAILED] {e}")

    # --------------------------------------------------
    # Strategy Narration
    # --------------------------------------------------
    def generate(self, prompt: str) -> str:
        if OPENAI_OFFLINE:
            return (
                "[MOCK MESSAGE]\n"
                "당신의 라이프스타일과 브랜드 톤에 맞춘 메시지를 전달합니다."
            )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a senior marketing copywriter. "
                            "Generate message only."
                        )
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"[OpenAI generate() FAILED] {e}")