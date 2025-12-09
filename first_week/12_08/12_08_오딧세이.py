import asyncio
import csv
import re
from pathlib import Path
from urllib.parse import urljoin
from playwright.async_api import async_playwright

# =========================
# 설정 — 브랜드 URL만 바꾸면 됨
# =========================
BRAND_URL = "https://www.amoremall.com/kr/ko/display/brand/detail/all?brandSn=16"  # ← 마몽드 예시

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_CSV = BASE_DIR / "data_csv" / "아모레" / "브랜드_세부.csv"


# =========================
# ★ 상품명 정제 — 프로모션 제거 기능
# =========================
def clean_name(name: str):
    # ★아세페특가★, ★ONLY특가★ 등 제거
    name = re.sub(r"★.*?★", "", name)
    # [싱글], [BEST], [대용량], [한정기획] 등 제거
    name = re.sub(r"\[.*?\]", "", name)
    # 불필요한 공백 제거
    return name.strip()


# =========================
# 팝업 닫힐 때까지 대기
# =========================
async def wait_for_popup_close(page):
    selector = ".popup, .layerPopup, .modal"
    if await page.locator(selector).count() == 0:
        return
    print("⛔ 팝업 감지됨 — 사람이 직접 닫을 때까지 대기 중…")

    while True:
        if await page.locator(selector).count() == 0:
            break
        await page.wait_for_timeout(300)

    print("👉 팝업 닫힘 — 계속 진행")


# =========================
# 리스트 카드 텍스트 파싱 (상품명 + 가격/할인/평점)
# =========================
def parse_card_text(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # 한글 포함 여부
    def has_korean(s: str) -> bool:
        return re.search(r"[가-힣]", s) is not None

    # 평점 단독 라인 (4.7 같은 것) 여부
    def is_score_line(s: str) -> bool:
        return re.fullmatch(r"\d\.\d", s) is not None

    # 리뷰 개수 단독 라인: (9,608) 같은 형태
    def is_review_count_line(s: str) -> bool:
        return re.fullmatch(r"\(\d[\d,]*\)", s) is not None

    # 1차: 한글이 들어간 줄 중에서 가장 긴 줄을 상품명으로 사용
    korean_lines = [ln for ln in lines if has_korean(ln)]
    if korean_lines:
        raw_name = max(korean_lines, key=len)
    else:
        # 2차 fallback: 가격·할인·URL·평점·리뷰개수 제거 후 첫 줄
        name_candidates = [
            ln
            for ln in lines
            if "원" not in ln
            and "%" not in ln
            and not ln.startswith("http")
            and not is_score_line(ln)
            and not is_review_count_line(ln)
        ]
        raw_name = name_candidates[0] if name_candidates else (lines[0] if lines else "")

    name = clean_name(raw_name)

    discount = ""
    price = ""
    rating = ""

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
# 상세 페이지: 용량
# =========================
async def extract_capacity(page):
    try:
        label = page.locator(
            "xpath=//*[contains(., '내용물의 용량') or contains(., '내용물의 중량')]"
        ).first
        await label.wait_for(timeout=10000)

        xpaths = [
            "xpath=//*[contains(., '내용물의 용량')]/following-sibling::*[1]",
            "xpath=//*[contains(., '내용물의 용량')]/parent::*[1]/following-sibling::*[1]",
        ]

        for xp in xpaths:
            block = page.locator(xp).first
            if await block.count() > 0:
                txt = (await block.inner_text()).strip()
                return " ".join(t.strip() for t in txt.splitlines() if t.strip())

        return ""
    except:
        return ""


# =========================
# 상세 페이지: 전성분
# =========================
async def extract_ingredients(page):
    try:
        header = page.locator(
            "xpath=//*[contains(., '화장품법') and contains(., '모든 성분')]"
        ).first
        await header.wait_for(timeout=15000)

        xpaths = [
            "xpath=//*[contains(., '모든 성분')]/following-sibling::*[1]",
            "xpath=//dt[contains(., '화장품법')]/following-sibling::dd[1]",
        ]

        for xp in xpaths:
            block = page.locator(xp).first
            if await block.count() > 0:
                text = (await block.inner_text()).strip()
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                return " ".join(lines)

        return ""
    except:
        return ""


# =========================
# 상세 페이지: 평점
# =========================
async def extract_rating(page):
    try:
        sel = page.locator("span:has-text('★') + span")
        if await sel.count() == 0:
            sel = page.locator("span.star-num, span.grade")

        if await sel.count() > 0:
            txt = (await sel.inner_text()).strip()
            m = re.search(r"\d\.\d", txt)
            return m.group(0) if m else ""
        return ""
    except:
        return ""


# =========================
# 고시 열기
# =========================
async def open_info_modal(page):
    selectors = [
        "text=상품정보제공 고시 보기",
        "button:has-text('상품정보제공 고시 보기')",
        ".btnViewMore",
    ]
    for sel in selectors:
        try:
            btn = page.locator(sel).first
            if await btn.count() > 0:
                await btn.click()
                break
        except:
            continue

    for _ in range(10):
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

        print("👉 브랜드 페이지 접속:", BRAND_URL)
        await page.goto(BRAND_URL, timeout=0)

        # 전체 스크롤 로딩
        last_height = None
        while True:
            height = await page.evaluate("() => document.body.scrollHeight")
            if height == last_height:
                break
            last_height = height
            await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(800)

        # 상품 카드 수집
        cards = page.locator("a[href*='/product/detail']")
        count = await cards.count()
        print(f"👉 감지된 상품 카드: {count}개")

        # URL 중복 제거
        products = {}
        for i in range(count):
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

            name, price, discount, rating_list = parse_card_text(text)

            products[pid] = {
                "url": url,
                "name": name,
                "price": price,
                "discount": discount,
                "rating": rating_list,
            }

        print(f"👉 최종 수집 대상: {len(products)}개")

        # 상세페이지 크롤링
        results = []

        for idx, (pid, info) in enumerate(products.items(), start=1):
            print(f"\n====== {idx}/{len(products)} ======")
            print(info["name"])
            print(info["url"])

            await page.goto(info["url"], timeout=0)
            await wait_for_popup_close(page)

            await open_info_modal(page)

            capacity = await extract_capacity(page)
            ingredients = await extract_ingredients(page)
            rating_detail = await extract_rating(page)

            results.append({
                "상품명": info["name"],
                "가격": info["price"],
                "할인율": info["discount"],
                "평점": rating_detail or info["rating"],
                "용량": capacity,
                "전성분": ingredients,
                "URL": info["url"],
            })

        await browser.close()

    # CSV 저장
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["상품명", "가격", "할인율", "평점", "용량", "전성분", "URL"],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n🎉 CSV 생성 완료!")
    print("📁 저장 경로:", OUTPUT_CSV)


if __name__ == "__main__":
    asyncio.run(main())