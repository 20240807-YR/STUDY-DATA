import asyncio
import csv
import re
from pathlib import Path
from urllib.parse import urljoin
from playwright.async_api import async_playwright

# =========================
# 설정
# =========================
BRAND_URL = "https://www.amoremall.com/kr/ko/display/brand/detail/all?brandSn=207"
CONCURRENCY = 10

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_CSV = BASE_DIR / "data_csv" / "아모레" / "브랜드_세부.csv"


# =========================
# 상품명 정제
# =========================
def clean_name(name: str):
    name = re.sub(r"★.*?★", "", name)
    name = re.sub(r"\[.*?]", "", name)
    return name.strip()


# =========================
# 카드 텍스트 파싱
# =========================
def parse_card_text(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    def has_kr(s): return re.search(r"[가-힣]", s)

    korean_lines = [ln for ln in lines if has_kr(ln)]
    raw_name = max(korean_lines, key=len) if korean_lines else lines[0]
    name = clean_name(raw_name)

    discount = price = rating = ""

    for ln in lines:
        if "%" in ln and not discount:
            m = re.search(r"\d+%", ln)
            if m:
                discount = m.group(0)

        if "원" in ln and not price:
            m = re.search(r"[\d,]+원", ln)
            if m:
                price = m.group(0)

        if not rating:
            m = re.search(r"\d\.\d", ln)
            if m:
                rating = m.group(0)

    return name, price, discount, rating


# =========================
# 상세: 원재료명 및 함량 (207 구조)
# =========================
async def extract_raw_materials(page):
    try:
        block = page.locator(
            "xpath=//h4[contains(text(), '원재')]/following-sibling::*[1]"
        ).first
        txt = await block.text_content()
        return " ".join(t.strip() for t in txt.splitlines() if t.strip()) if txt else ""
    except:
        return ""


# =========================
# 상세: 영양성분 (207 구조)
# =========================
async def extract_nutrition(page):
    try:
        block = page.locator(
            "xpath=//h4[contains(text(), '영양성분')]/following-sibling::*[1]"
        ).first
        txt = await block.text_content()
        return " ".join(t.strip() for t in txt.splitlines() if t.strip()) if txt else ""
    except:
        return ""


# =========================
# 상세: 용량
# =========================
async def extract_capacity(page):
    try:
        block = page.locator(
            "xpath=//h4[contains(text(), '용량') or contains(text(),'중량')]/following-sibling::*[1]"
        ).first
        txt = await block.text_content()
        return " ".join(t.strip() for t in txt.splitlines() if t.strip()) if txt else ""
    except:
        return ""


# =========================
# 상세: 평점
# =========================
async def extract_rating(page):
    try:
        sel = page.locator("span.star-num, span.grade, span:has-text('★') + span")
        if await sel.count() > 0:
            txt = await sel.text_content()
            m = re.search(r"\d\.\d", txt)
            return m.group(0) if m else ""
        return ""
    except:
        return ""


# =========================
# 상세페이지 병렬 처리
# =========================
async def scrape_detail(browser, info, idx, total):
    page = await browser.new_page()

    try:
        print(f"[{idx}/{total}] 상세: {info['name']}")

        await page.goto(info["url"], timeout=0)
        await page.wait_for_timeout(500)

        capacity = await extract_capacity(page)
        raw_materials = await extract_raw_materials(page)
        nutrition = await extract_nutrition(page)
        rating = await extract_rating(page)

        return {
            "상품명": info["name"],
            "가격": info["price"],
            "할인율": info["discount"],
            "평점": rating or info["rating"],
            "용량": capacity,
            "원재료명 및 함량": raw_materials,
            "영양성분": nutrition,
            "URL": info["url"],
        }

    finally:
        await page.close()


# =========================
# 메인
# =========================
async def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        print("👉 브랜드 페이지 열기")
        await page.goto(BRAND_URL, timeout=0)

        last = None
        while True:
            h = await page.evaluate("() => document.body.scrollHeight")
            if h == last:
                break
            last = h
            await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(500)

        cards = page.locator("a[href*='/product/detail']")
        count = await cards.count()
        print(f"👉 감지된 상품 수: {count}")

        products = {}
        for i in range(count):
            href = await cards.nth(i).get_attribute("href")
            if not href:
                continue

            m = re.search(r"onlineProdSn=(\d+)", href)
            if not m:
                continue

            pid = m.group(1)

            url = urljoin(BRAND_URL, href)
            raw = await cards.nth(i).inner_text()
            name, price, discount, rating = parse_card_text(raw)

            products[pid] = {
                "url": url,
                "name": name,
                "price": price,
                "discount": discount,
                "rating": rating,
            }

        products = list(products.values())
        total = len(products)
        print(f"👉 최종 대상: {total}개")

        sem = asyncio.Semaphore(CONCURRENCY)

        async def limited_task(idx, info):
            async with sem:
                return await scrape_detail(browser, info, idx, total)

        tasks = [limited_task(i + 1, info) for i, info in enumerate(products)]
        results = await asyncio.gather(*tasks)

        await browser.close()

    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "상품명", "가격", "할인율", "평점",
                "용량", "원재료명 및 함량", "영양성분", "URL"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n🎉 CSV 저장 완료")
    print("📁 위치:", OUTPUT_CSV)


if __name__ == "__main__":
    asyncio.run(main())