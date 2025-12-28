class MarketContextTool:
    """
    Optional Market Context Tool (mock)
    """

    def run(self, crm_row: dict) -> dict:
        return {
            "brand": crm_row.get("brand"),
            "tone_cluster": crm_row.get("brand_tone_cluster"),
            "trend_summary": (
                "최근 해당 톤의 브랜드는 감성 중심 메시지에 대한 "
                "고객 반응률이 상승하는 경향을 보임"
            ),
            "confidence": 0.63,
            "source": "mock_market_context",
        }