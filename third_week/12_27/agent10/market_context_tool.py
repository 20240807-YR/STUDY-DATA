# agent10/market_context_tool.py
# ============================================================
# Market Context Tool (Optional)
# - 외부 시장/트렌드 정보 보강용
# - 현재는 mock 구현 (제출용)
# ============================================================

class MarketContextTool:
    """
    Market Context Tool

    역할:
    - CRM 전략 설명 시 외부 맥락(시장/톤 트렌드)을 보강
    - LLM의 '추천/계산'을 대체하지 않음
    - Optional Tool (Controller에서 선택적으로 활성화)

    확장:
    - SerpAPIWrapper
    - 브랜드/카테고리 트렌드 검색
    """

    def run(self, crm_row: dict) -> dict:
        brand = crm_row.get("brand")
        tone_cluster = crm_row.get("brand_tone_cluster")

        # 현재는 mock 결과 (제출용)
        return {
            "brand": brand,
            "tone_cluster": tone_cluster,
            "trend_summary": (
                "최근 해당 톤의 브랜드는 감성 중심 메시지에 대한 "
                "고객 반응률이 상승하는 경향을 보임"
            ),
            "confidence": 0.63,
            "source": "mock_market_context"
        }