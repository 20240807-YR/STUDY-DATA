import asyncio
import csv
import re
from pathlib import Path
from urllib.parse import urljoin
from playwright.async_api import async_playwright

# 홀리추얼 브랜드 페이지
BRAND_URL = "https://www.amoremall.com/kr/ko/display/brand/detail?brandSn=27"

# CSV 저장 경로
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_CSV = BASE_DIR / "data_csv" / "아모레" / "브랜드_세부.csv"


# =========================
# 공통 텍스트 정리
# =========================
def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u00a0", " ")
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return " ".join(lines)


# =========================
# 메인 페이지 카드 텍스트 → 이름/가격/할인/평점
# (홀리추얼 전용)
# =========================
def parse_card_text(text: str):
    txt = clean_text(text)

    price = ""
    discount = ""
    rating = ""

    # 가격
    m = re.search(r"([\d,]+\s*원)", txt)
    if m:
        price = m.group(1).strip()
        txt = txt.replace(m.group(1), " ")

    # 할인율
    m = re.search(r"(\d+%)", txt)
    if m:
        discount = m.group(1)
        txt = txt.replace(m.group(1), " ")

    # 평점
    m = re.search(r"(\d\.\d)", txt)
    if m:
        rating = m.group(1)
        txt = txt.replace(m.group(1), " ")

    # ( 1,322 ) 이런 리뷰수, '좋아요' 제거
    txt = re.sub(r"\(\s*[\d,]+\s*\)", " ", txt)
    txt = txt.replace("좋아요", " ")

    # 앞뒤 남은 텍스트 = 상품명 (+ 용량 정도)
    name = re.sub(r"\s{2,}", " ", txt).strip()

    return name, price, discount, rating


# =========================
# 상세페이지에서 정보 추출
# =========================
async def extract_product_detail(browser, url: str, list_name: str, list_price: str,
                                list_discount: str, list_rating: str):
    page = await browser.new_page()
    try:
        print("  ▶ 상세 페이지 이동:", url)
        await page.goto(url, timeout=0)
        await page.wait_for_timeout(900)

        # ---------- 상품명 (상세 기준으로 다시 시도) ----------
        name = ""
        name_selectors = [
            "h1",
            ".product-name",
            ".prd_name",
            ".prod_title",
            ".tit",
            ".title",
            "[itemprop='name']",
        ]
        for sel in name_selectors:
            el = page.locator(sel).first
            if await el.count() > 0:
                txt = clean_text(await el.inner_text())
                if txt:
                    name = txt
                    break

        # 그래도 못 찾으면 <title> 사용
        if not name:
            try:
                title_txt = await page.title()
                name = title_txt.split("|")[0].strip()
            except:
                name = ""

        # 그래도 비면 메인 카드에서 가져온 이름 사용
        if not name:
            name = list_name

        # ---------- 가격 ----------
        price = list_price
        if not price:
            price_selectors = [
                ".price",
                ".amount",
                ".prod-price",
                "span:has-text('원')",
            ]
            for sel in price_selectors:
                el = page.locator(sel).first
                if await el.count() > 0:
                    txt = clean_text(await el.inner_text())
                    m = re.search(r"[\d,]+원", txt)
                    if m:
                        price = m.group(0)
                        break

        # ---------- 할인율 ----------
        discount = list_discount
        if not discount:
            disc_selectors = [
                ".discount",
                ".sale",
                ".rate",
                "span:has-text('%')",
            ]
            for sel in disc_selectors:
                el = page.locator(sel).first
                if await el.count() > 0:
                    txt = clean_text(await el.inner_text())
                    m = re.search(r"\d+%", txt)
                    if m:
                        discount = m.group(0)
                        break

        # ---------- 평점 ----------
        rating = list_rating
        if not rating:
            r_el = page.locator(".star-num, .grade").first
            if await r_el.count() > 0:
                txt = clean_text(await r_el.inner_text())
                m = re.search(r"\d\.\d", txt)
                if m:
                    rating = m.group(0)

        # ---------- 상품정보제공 고시 보기 ----------
        info_opened = False
        info_selectors = [
            "text=상품정보제공 고시 보기",
            "button:has-text('상품정보제공 고시 보기')",
            "text=상품정보제공고시",
            ".btnViewMore",
        ]

        for sel in info_selectors:
            btn = page.locator(sel).first
            if await btn.count() > 0:
                try:
                    await btn.click()
                    await page.wait_for_timeout(800)
                    info_opened = True
                    break
                except:
                    pass

        # 안 뜨면 내려가면서 한 번 더 시도
        if not info_opened:
            for _ in range(5):
                await page.mouse.wheel(0, 1200)
                await page.wait_for_timeout(300)
            for sel in info_selectors:
                btn = page.locator(sel).first
                if await btn.count() > 0:
                    try:
                        await btn.click()
                        await page.wait_for_timeout(800)
                        info_opened = True
                        break
                    except:
                        pass

        capacity = ""
        ingredients = ""

        if info_opened:
            # ---------- 용량 / 중량 ----------
            cap_nodes = page.locator(
                "xpath=//*[contains(text(),'내용물의 용량') "
                "or contains(text(),'내용물의 중량') "
                "or contains(text(),'용량')]"
            )
            if await cap_nodes.count() > 0:
                cap = cap_nodes.first
                sib = cap.locator("xpath=following-sibling::*[1]").first
                if await sib.count() == 0:
                    sib = cap.locator("xpath=parent::*/following-sibling::*[1]").first
                if await sib.count() > 0:
                    capacity = clean_text(await sib.inner_text())

            # ---------- 전성분 (화장품법 / 전성분) ----------
            ing_nodes = page.locator(
                "xpath=//*[contains(text(),'화장품법') or contains(text(),'전성분')]"
            )
            if await ing_nodes.count() > 0:
                ing = ing_nodes.first
                sib = ing.locator("xpath=following-sibling::*[1]").first
                if await sib.count() == 0:
                    sib = ing.locator("xpath=parent::*/following-sibling::*[1]").first
                if await sib.count() > 0:
                    ingredients = clean_text(await sib.inner_text())

        return {
            "상품명": name,
            "가격": price,
            "할인율": discount,
            "평점": rating,
            "용량": capacity,
            "전성분": ingredients,
            "URL": url,
        }

    finally:
        await page.close()


# =========================
# 메인
# =========================
async def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        print("👉 브랜드 페이지 접속:", BRAND_URL)
        await page.goto(BRAND_URL, timeout=0)

        # 홀리추얼은 전체상품이 한 페이지에 거의 다 있어서
        # 안전하게 아래까지 한 번 내려주고 끝낸다
        print("👉 전체 상품 스크롤 한 번 진행...")
        last_height = -1
        while True:
            height = await page.evaluate("() => document.body.scrollHeight")
            if height == last_height:
                break
            last_height = height
            await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(800)
        print("✔ 스크롤 완료")

        # -------- 카드들에서 href + 텍스트 수집 --------
        cards = page.locator("a[href*='/product/detail']")
        count = await cards.count()
        print(f"👉 감지된 a태그 카드: {count}개")

        products = {}  # onlineProdSn 기준으로 중복 제거

        for i in range(count):
            href = await cards.nth(i).get_attribute("href")
            if not href:
                continue

            m = re.search(r"onlineProdSn=(\d+)", href)
            if not m:
                continue

            pid = m.group(1)

            text = await cards.nth(i).inner_text()
            name, price, discount, rating = parse_card_text(text)

            # ❗ 한글이 없는 텍스트(상세보기 버튼 등)는 건너뜀
            if not re.search(r"[가-힣]", name):
                continue

            # 이제부터는 '제대로 된 카드'만 pid 등록
            if pid in products:
                continue

            url = urljoin(BRAND_URL, href)

            products[pid] = {
                "url": url,
                "name": name,
                "price": price,
                "discount": discount,
                "rating": rating,
            }

        print(f"👉 최종 수집 대상(중복 제거 후): {len(products)}개")

        # -------- 상세페이지 크롤링 --------
        results = []
        items = list(products.items())

        for idx, (pid, info) in enumerate(items, start=1):
            print(f"\n===== {idx}/{len(items)} =====")
            print("카드 상품명:", info["name"])
            print("URL:", info["url"])

            detail = await extract_product_detail(
                browser,
                info["url"],
                info["name"],
                info["price"],
                info["discount"],
                info["rating"],
            )
            results.append(detail)

        await browser.close()

    # -------- CSV 저장 --------
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["상품명", "가격", "할인율", "평점", "용량", "전성분", "URL"],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n🎉 CSV 저장 완료!")
    print("📁 저장 경로:", OUTPUT_CSV)


if __name__ == "__main__":
    asyncio.run(main())