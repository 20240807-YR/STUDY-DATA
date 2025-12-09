import asyncio
import csv
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs
from playwright.async_api import async_playwright

BRAND_URL = "https://www.amoremall.com/kr/ko/display/brand/detail/all?brandSn=204"

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_CSV = BASE_DIR / "data_csv" / "아모레" / "브랜드_세부.csv"


def clean_name(name: str):
    name = re.sub(r"★.*?★", "", name)
    name = re.sub(r"\[.*?\]", "", name)
    return name.strip()


def normalize_detail_url(url: str):
    """
    브랜드 페이지에서 가끔 brand.amoremall.com 형태로 나오는 링크가 있어
    이를 www.amoremall.com의 정규 상세페이지 URL로 강제 변환하는 함수.
    """
    if "brand.amoremall.com" in url:
        qs = parse_qs(urlparse(url).query)
        if "onlineProdSn" in qs:
            prod = qs["onlineProdSn"][0]
            return f"https://www.amoremall.com/kr/ko/product/detail?onlineProdSn={prod}"
    return url


def parse_card_text(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    def has_korean(s): return re.search(r"[가-힣]", s)
    korean_lines = [ln for ln in lines if has_korean(ln)]
    raw_name = max(korean_lines, key=len) if korean_lines else lines[0]
    name = clean_name(raw_name)

    discount = price = rating = ""
    for ln in lines:
        if "%" in ln and not discount:
            m = re.search(r"\d+%", ln)
            if m: discount = m.group(0)
        if "원" in ln and not price:
            m = re.search(r"[\d,]+원", ln)
            if m: price = m.group(0)
        if not rating:
            m = re.search(r"\d\.\d", ln)
            if m: rating = m.group(0)
    return name, price, discount, rating


async def wait_for_popup_close(page):
    selector = ".popup, .layerPopup, .modal"
    while True:
        if await page.locator(selector).count() == 0:
            break
        await page.wait_for_timeout(300)


async def open_info_modal(page):
    selectors = [
        "text=상품정보제공 고시 보기",
        "button:has-text('상품정보제공 고시 보기')",
        ".btnViewMore",
    ]
    for sel in selectors:
        btn = page.locator(sel).first
        if await btn.count() > 0:
            await btn.click()
            await page.wait_for_timeout(800)
            return True
    return False


async def extract_capacity(page):
    sel = page.locator("xpath=//*[contains(text(),'내용물의 용량') or contains(text(),'내용물의 중량')]")
    if await sel.count() > 0:
        sib = sel.first.locator("xpath=following-sibling::*[1]")
        if await sib.count() > 0:
            return (await sib.inner_text()).strip()
    return ""


async def extract_ingredients(page):
    sel = page.locator("xpath=//*[contains(text(),'모든 성분') or contains(text(),'화장품법')]")
    if await sel.count() > 0:
        sib = sel.first.locator("xpath=following-sibling::*[1]")
        if await sib.count() > 0:
            txt = (await sib.inner_text()).strip()
            return " ".join([ln.strip() for ln in txt.splitlines() if ln.strip()])
    return ""


async def extract_rating(page):
    r = page.locator(".star-num, .grade")
    if await r.count() > 0:
        txt = await r.first.inner_text()
        m = re.search(r"\d\.\d", txt)
        if m:
            return m.group(0)
    return ""


async def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        await page.goto(BRAND_URL, timeout=0)

        # 스크롤 로딩
        while True:
            h1 = await page.evaluate("() => document.body.scrollHeight")
            await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(600)
            h2 = await page.evaluate("() => document.body.scrollHeight")
            if h1 == h2:
                break

        cards = page.locator("a[href*='/product/detail']")
        count = await cards.count()

        items = []
        for i in range(count):
            href = await cards.nth(i).get_attribute("href")
            if not href:
                continue
            url = normalize_detail_url(urljoin(BRAND_URL, href))
            text = await cards.nth(i).inner_text()
            name, price, discount, rating = parse_card_text(text)

            items.append({
                "url": url,
                "name": name,
                "price": price,
                "discount": discount,
                "rating": rating,
            })

        results = []

        for idx, item in enumerate(items, start=1):
            detail = await browser.new_page()
            await detail.goto(item["url"], timeout=0)
            await wait_for_popup_close(detail)

            opened = await open_info_modal(detail)

            capacity = await extract_capacity(detail) if opened else ""
            ingredients = await extract_ingredients(detail) if opened else ""
            rating_detail = await extract_rating(detail)

            results.append({
                "상품명": item["name"],
                "가격": item["price"],
                "할인율": item["discount"],
                "평점": rating_detail or item["rating"],
                "용량": capacity,
                "전성분": ingredients,
                "URL": item["url"],
            })

            await detail.close()

        await browser.close()

    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["상품명", "가격", "할인율", "평점", "용량", "전성분", "URL"],
        )
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    asyncio.run(main())