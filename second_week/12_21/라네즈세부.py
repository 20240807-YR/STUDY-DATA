import asyncio
import csv
import re
from pathlib import Path
from urllib.parse import urljoin
from playwright.async_api import async_playwright

# =========================
# 설정
# =========================
BRAND_NAME = "롱테이크"
BRAND_URL = "https://www.amoremall.com/kr/ko/display/brand/detail/all?brandSn=197"

BASE_DIR = Path.cwd()
OUTPUT_CSV = BASE_DIR / "brand_tone_text" / f"{BRAND_NAME}_raw_text.csv"


# =========================
# 제품 설명 텍스트 추출
# =========================
async def extract_product_text(page):
    selectors = [
        "div.prd-desc",
        "div.prd-detail",
        "section:has-text('상품 설명')",
        "section:has-text('효능')",
        "section:has-text('사용 방법')"
    ]

    texts = []
    for sel in selectors:
        loc = page.locator(sel)
        if await loc.count() > 0:
            txt = (await loc.first.inner_text()).strip()
            if txt and len(txt) > 30:
                texts.append(txt)

    return " ".join(texts)


# =========================
# 메인
# =========================
async def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        print("👉 브랜드 페이지 접속")
        await page.goto(BRAND_URL, timeout=0)

        # 무한 스크롤 로딩
        last_height = None
        while True:
            height = await page.evaluate("() => document.body.scrollHeight")
            if height == last_height:
                break
            last_height = height
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(800)

        cards = page.locator("a[href*='/product/detail']")
        count = await cards.count()
        print(f"👉 감지된 제품 카드: {count}개")

        products = {}
        for i in range(count):
            href = await cards.nth(i).get_attribute("href")
            if not href:
                continue

            m = re.search(r"onlineProdSn=(\d+)", href)
            if not m:
                continue

            prod_id = m.group(1)
            if prod_id in products:
                continue

            url = urljoin(BRAND_URL, href)
            name = (await cards.nth(i).inner_text()).splitlines()[0].strip()

            products[prod_id] = {
                "name": name,
                "url": url
            }

        print(f"👉 실제 수집 대상: {len(products)}개")

        results = []

        for idx, info in enumerate(products.values(), start=1):
            print(f"\n[{idx}/{len(products)}] {info['name']}")
            await page.goto(info["url"], timeout=0)
            await page.wait_for_timeout(1200)

            raw_text = await extract_product_text(page)

            results.append({
                "product_name": info["name"],
                "brand": BRAND_NAME,
                "raw_text": raw_text,
                "product_url": info["url"]
            })

        await browser.close()

    # CSV 저장
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["product_name", "brand", "raw_text", "product_url"]
        )
        writer.writeheader()
        writer.writerows(results)

    print("🎉 전량 상세페이지 텍스트 수집 완료")
    print("📁 저장 위치:", OUTPUT_CSV)


if __name__ == "__main__":
    asyncio.run(main())