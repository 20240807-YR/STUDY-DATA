import asyncio
import csv
import re
from pathlib import Path
from urllib.parse import urljoin
from playwright.async_api import async_playwright

# =========================
# 설정
# =========================
BRAND_URL = "https://www.amoremall.com/kr/ko/display/brand/detail/all?brandSn=197"

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_CSV = BASE_DIR / "data_csv" / "아모레" / "롱테이크_세부.csv"


# =========================
# 팝업 수동 닫기 대기
# =========================
async def wait_for_manual_popup_close(page):
    popup_selector = ".popup, .layerPopup, .modal"
    if await page.locator(popup_selector).count() == 0:
        return
    print("⛔ 팝업 감지됨 — 사람이 직접 닫을 때까지 대기 중…")
    while True:
        if await page.locator(popup_selector).count() == 0:
            break
        await page.wait_for_timeout(300)
    print("👉 팝업 닫힘 — 다음 단계")


# =========================
# 리스트 카드 파싱
# =========================
def parse_card_text(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    name_candidates = [ln for ln in lines if "원" not in ln and "%" not in ln]
    name = name_candidates[0] if name_candidates else (lines[0] if lines else "")

    discount, price, rating = "", "", ""
    for ln in lines:
        if not discount and "%" in ln:
            m = re.search(r"-?\d+%|\d+%", ln)
            if m:
                discount = m.group(0)
        if not price and "원" in ln:
            m = re.search(r"[\d,]+원", ln)
            if m:
                price = m.group(0)
        if not rating:
            m = re.search(r"\d\.\d", ln)
            if m:
                rating = m.group(0)
    return name, price, discount, rating


# =========================
# 상세 페이지 정보 추출
# =========================
async def extract_capacity(page):
    try:
        label = page.locator(
            "xpath=//*[contains(., '내용물의 용량') or contains(., '내용물의 중량')]"
        ).first
        await label.wait_for(timeout=10000)
        xps = [
            "xpath=//*[contains(., '내용물의 용량') or contains(., '내용물의 중량')]/following-sibling::*[1]",
            "xpath=//*[contains(., '내용물의 용량') or contains(., '내용물의 중량')]/parent::*[1]/following-sibling::*[1]",
        ]
        for xp in xps:
            block = page.locator(xp).first
            if await block.count() > 0:
                txt = (await block.inner_text()).strip()
                if txt:
                    return " ".join(t.strip() for t in txt.splitlines() if t.strip())
        return ""
    except:
        return ""


async def extract_ingredients(page):
    try:
        header = page.locator(
            "xpath=//*[contains(., '화장품법') and contains(., '모든 성분')]"
        ).first
        await header.wait_for(timeout=15000)

        xps = [
            "xpath=//*[contains(., '화장품법') and contains(., '모든 성분')]/following-sibling::*[1]",
            "xpath=//dt[contains(., '화장품법')]/following-sibling::dd[1]",
        ]
        for xp in xps:
            block = page.locator(xp).first
            if await block.count() > 0:
                text = (await block.inner_text()).strip()
                if text:
                    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                    return " ".join(lines)
        return ""
    except:
        return ""


async def extract_rating(page):
    try:
        rate = page.locator("span:has-text('★') + span").first
        if await rate.count() == 0:
            rate = page.locator("span.star-num, span.grade").first
        if await rate.count() > 0:
            txt = (await rate.inner_text()).strip()
            m = re.search(r"\d\.\d", txt)
            return m.group(0) if m else ""
        return ""
    except:
        return ""


async def open_info_modal(page):
    selectors = [
        "text=상품정보제공 고시 보기",
        "button:has-text('상품정보제공 고시 보기')",
        "button.btnViewMore",
        "div.linkArea button",
    ]
    for sel in selectors:
        try:
            btn = page.locator(sel).first
            if await btn.count() > 0:
                await btn.scroll_into_view_if_needed()
                await page.wait_for_timeout(300)
                await btn.click()
                break
        except:
            continue
    for _ in range(12):
        await page.mouse.wheel(0, 1200)
        await page.wait_for_timeout(250)


# =========================
# 메인
# =========================
async def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        print("👉 롱테이크 페이지 접속:", BRAND_URL)
        await page.goto(BRAND_URL, timeout=0)

        # 전체 스크롤 로딩
        last_height = None
        while True:
            height = await page.evaluate("() => document.body.scrollHeight")
            if height == last_height:
                break
            last_height = height
            await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(900)

        cards = page.locator("a[href*='/product/detail']")
        count = await cards.count()
        print(f"👉 감지된 상품 카드: {count}개")

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
            text = (await cards.nth(i).inner_text()).strip()
            name, price, discount, rating = parse_card_text(text)
            products[prod_id] = {
                "url": url,
                "name": name,
                "price": price,
                "discount": discount,
                "rating_list": rating,
            }

        print(f"👉 실제 수집 대상 상품 수: {len(products)}개")

        results = []
        for idx, (prod_id, info) in enumerate(products.items(), start=1):
            url = info["url"]
            print(f"\n====== {idx}/{len(products)} ======")
            print(info["name"], url)

            await page.goto(url, timeout=0)
            await wait_for_manual_popup_close(page)

            rating_detail = await extract_rating(page)
            rating_final = rating_detail or info["rating_list"]

            await open_info_modal(page)
            capacity = await extract_capacity(page)
            ingredients = await extract_ingredients(page)

            results.append({
                "상품명": info["name"],
                "가격": info["price"],
                "할인율": info["discount"],
                "평점": rating_final,
                "용량": capacity,
                "전성분": ingredients,
                "URL": url,
            })

        await browser.close()

    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "상품명", "가격", "할인율", "평점", "용량", "전성분", "URL"
        ])
        writer.writeheader()
        writer.writerows(results)

    print("🎉 CSV 생성 완료:", OUTPUT_CSV)

if __name__ == "__main__":
    asyncio.run(main())
    