import asyncio
import csv
import re
from pathlib import Path
from urllib.parse import urljoin
from playwright.async_api import async_playwright


# =========================
# 브랜드 URL — 여기만 바꾸면 됨
# =========================
BRAND_URL = "https://www.amoremall.com/kr/ko/display/brand/detail/all?brandSn=12"  # 바이탈뷰티


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_CSV = BASE_DIR / "data_csv" / "아모레" / "브랜드_세부.csv"


# =========================
# 상품명 클린 + 상품명 정확 검출
# =========================
def clean_name(name: str):
    name = re.sub(r"★.*?★", "", name)
    name = re.sub(r"\[.*?\]", "", name)
    return name.strip()


def extract_real_name(lines):
    cleaned = [ln.strip() for ln in lines if ln.strip()]

    filtered = []
    for ln in cleaned:
        if re.search(r"\d+원", ln):  # 가격 제거
            continue
        if re.search(r"\d+%", ln):  # 할인 제거
            continue
        if re.fullmatch(r"\d\.\d", ln):  # 평점 제거
            continue
        if ln in ["정직해요", "좋아요", "용량 넉넉해요", "향이 좋아요", "촉촉해요"]:
            continue
        filtered.append(ln)

    if not filtered:
        return cleaned[0] if cleaned else ""

    return max(filtered, key=len)


def parse_card_text(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    raw_name = extract_real_name(lines)
    name = clean_name(raw_name)

    discount = price = rating = ""

    for ln in lines:
        if not discount:
            m = re.search(r"-?\d+%", ln)
            if m:
                discount = m.group(0)

        if not price:
            m = re.search(r"[\d,]+원", ln)
            if m:
                price = m.group(0)

        if not rating:
            m = re.search(r"\d\.\d", ln)
            if m:
                rating = m.group(0)

    return name, price, discount, rating


# =========================
# 팝업 닫히는 거 기다리기
# =========================
async def wait_for_popup_close(page):
    selector = ".popup, .layerPopup, .modal"
    if await page.locator(selector).count() == 0:
        return
    print("⛔ 팝업 감지됨 — 닫을 때까지 대기 중…")
    while True:
        if await page.locator(selector).count() == 0:
            break
        await page.wait_for_timeout(300)
    print("👉 팝업 닫힘")


# =========================
# 상세페이지 — 용량
# =========================
async def extract_capacity(page):
    try:
        label = page.locator(
            "xpath=//*[contains(., '용량') or contains(., '중량')]"
        ).first
        await label.wait_for(timeout=10000)

        paths = [
            "xpath=//*[contains(., '용량')]/following-sibling::*[1]",
            "xpath=//*[contains(., '중량')]/following-sibling::*[1]",
        ]

        for xp in paths:
            block = page.locator(xp).first
            if await block.count():
                txt = (await block.inner_text()).strip()
                return " ".join(t.strip() for t in txt.splitlines() if t.strip())

        return ""
    except:
        return ""


# =========================
# 상세페이지 — 바이탈뷰티 전용 "원재료명 및 함량"
# =========================
async def extract_ingredients(page):
    try:
        header = page.locator(
            "xpath=//*[contains(., '원재료명및 함량') or contains(., '원재료명 및 함량')]"
        ).first
        await header.wait_for(timeout=15000)

        xps = [
            "xpath=//*[contains(., '원재료명및 함량')]/following-sibling::*[1]",
            "xpath=//*[contains(., '원재료명 및 함량')]/following-sibling::*[1]",
            "xpath=//dt[contains(., '원재료명')]/following-sibling::dd[1]",
        ]

        for xp in xps:
            block = page.locator(xp).first
            if await block.count():
                raw = (await block.inner_text()).strip()
                lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
                return " ".join(lines)

        return ""
    except:
        return ""


# =========================
# 상세페이지 — 평점
# =========================
async def extract_rating(page):
    try:
        sel = page.locator("span:has-text('★') + span")
        if not await sel.count():
            sel = page.locator("span.star-num, span.grade")

        if await sel.count():
            txt = (await sel.inner_text()).strip()
            m = re.search(r"\d\.\d", txt)
            return m.group(0) if m else ""

        return ""
    except:
        return ""


# =========================
# 상품정보 고시 열기
# =========================
async def open_info(page):
    candidates = [
        "text=상품정보제공 고시 보기",
        "button.btnViewMore",
        "button:has-text('상품정보제공 고시')",
    ]
    for sel in candidates:
        try:
            btn = page.locator(sel).first
            if await btn.count():
                await btn.click()
                break
        except:
            continue

    for _ in range(10):
        await page.mouse.wheel(0, 1200)
        await page.wait_for_timeout(250)


# =========================
# 메인 실행
# =========================
async def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)
        page = await browser.new_page()

        print("👉 브랜드 페이지 이동:", BRAND_URL)
        await page.goto(BRAND_URL, timeout=0)

        # 스크롤 끝까지
        last = None
        while True:
            h = await page.evaluate("() => document.body.scrollHeight")
            if h == last:
                break
            last = h
            await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(800)

        cards = page.locator("a[href*='/product/detail']")
        total = await cards.count()
        print(f"👉 감지된 상품 카드: {total}")

        products = {}

        for i in range(total):
            href = await cards.nth(i).get_attribute("href")
            if not href:
                continue

            m = re.search(r"onlineProdSn=(\d+)", href)
            if not m:
                continue

            pid = m.group(1)
            if pid in products:
                continue

            url = urljoin(BRAND_URL, href)
            text = await cards.nth(i).inner_text()

            name, price, discount, rating = parse_card_text(text)

            products[pid] = {
                "url": url,
                "name": name,
                "price": price,
                "discount": discount,
                "rating": rating,
            }

        print(f"👉 최종 수집 대상 상품 수: {len(products)}")

        results = []

        for idx, (pid, info) in enumerate(products.items(), start=1):
            print(f"\n====== {idx}/{len(products)} ======")
            print(info["name"])
            print(info["url"])

            await page.goto(info["url"], timeout=0)
            await wait_for_popup_close(page)
            await open_info(page)

            capacity = await extract_capacity(page)
            ingredients = await extract_ingredients(page)
            rating_detail = await extract_rating(page)

            results.append({
                "상품명": info["name"],
                "가격": info["price"],
                "할인율": info["discount"],
                "평점": rating_detail or info["rating"],
                "용량": capacity,
                "원재료명및함량": ingredients,
                "URL": info["url"],
            })

        await browser.close()

    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "상품명",
                "가격",
                "할인율",
                "평점",
                "용량",
                "원재료명및함량",
                "URL",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n🎉 CSV 생성 완료:", OUTPUT_CSV)


if __name__ == "__main__":
    asyncio.run(main())