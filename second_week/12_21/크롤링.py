# -*- coding: utf-8 -*-
"""
APGroup Brands -> 각 브랜드 상세(AP intro) + 공식사이트(브랜드 스토리) 크롤링
- 브랜드마다 반드시 https://www.apgroup.com/int/ko/brands/brands.html 를 거쳐서 진입/복귀
- 결과: brand_identity_txt/{brand_slug}.txt
- Playwright (async) 필요
"""

import asyncio
import os
import re
from urllib.parse import urljoin, urlparse

from playwright.async_api import async_playwright, TimeoutError as PWTimeoutError


AP_BRANDS_MAIN = "https://www.apgroup.com/int/ko/brands/brands.html"
OUT_DIR = "brand_identity_txt"


# ---------------------------
# 텍스트 정리(제품/쇼핑몰 잡음 제거)
# ---------------------------
NOISE_PATTERNS = [
    r"\{\$\*.*?\}",                   # {$*...} 템플릿
    r"^\s*(SOLD\s*OUT|BEST|NEW)\s*$",
    r"리뷰\s*\d+",
    r"평점",
    r"배송비",
    r"쿠폰",
    r"장바구니",
    r"구매하기",
    r"회원가입",
    r"로그인",
    r"상품(요약정보|코드|문의|자유게시판)?",
    r"소비자가",
    r"판매가",
    r"최대혜택가",
    r"원산지",
    r"제조사",
    r"공급사",
    r"사업자번호",
    r"통신판매업",
    r"에스크로",
    r"카페24|CAFE24",
    r"개인정보\s*처리방침",
    r"이용약관",
    r"고객센터",
    r"처리중입니다",
    r"검색",
    r"추천\s*검색어",
    r"기획전|이벤트|공지사항|FAQ|문의",
    r"^#\s*.+",                        # 해시태그
    r"^\s*\d{2,3}(,\d{3})*\s*$",       # 가격 숫자만
    r"^\s*\d+\s*$",
]

NOISE_RE = re.compile("|".join(f"(?:{p})" for p in NOISE_PATTERNS), re.IGNORECASE)

# 너무 쇼핑몰 냄새 강한 라인(가격/할인/퍼센트 등)
SHOP_HEAVY_RE = re.compile(r"(\d{1,3}%|\bUP\s*TO\b|,?\d{3,}원|P\d{5,})", re.IGNORECASE)

def clean_text(raw: str) -> str:
    # 기본 정리
    raw = raw.replace("\r", "\n")
    lines = [ln.strip() for ln in raw.split("\n")]
    cleaned = []
    prev = ""
    for ln in lines:
        if not ln:
            continue
        if ln == prev:  # 반복 제거
            continue
        prev = ln

        # 너무 짧은 메뉴성 라인 다수 제거(단, 한글 제목은 남길 수 있게 완전 짧은 것만)
        if len(ln) <= 2:
            continue

        # 잡음 패턴 제거
        if NOISE_RE.search(ln):
            continue
        if SHOP_HEAVY_RE.search(ln) and len(ln) < 60:
            continue

        cleaned.append(ln)

    # 과도한 공백 정리
    out = "\n".join(cleaned)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


# ---------------------------
# 팝업/동의/배너 닫기(가능한 것만)
# ---------------------------
POPUP_BUTTON_TEXTS = [
    "닫기", "오늘 하루 보지않기", "거부하기", "거부", "동의하지 않음",
    "동의하고", "동의", "확인", "나중에", "취소",
]

async def try_close_popups(page):
    for _ in range(5):
        closed_any = False
        for t in POPUP_BUTTON_TEXTS:
            loc = page.get_by_text(t, exact=False)
            try:
                if await loc.count() > 0:
                    # 보이는 것만 누르기
                    for i in range(min(await loc.count(), 3)):
                        item = loc.nth(i)
                        if await item.is_visible():
                            try:
                                await item.click(timeout=1200)
                                closed_any = True
                                await page.wait_for_timeout(300)
                            except Exception:
                                # 클릭 안 되면 JS click 시도
                                try:
                                    h = await item.element_handle()
                                    if h:
                                        await page.evaluate("(el)=>el.click()", h)
                                        closed_any = True
                                        await page.wait_for_timeout(300)
                                except Exception:
                                    pass
            except Exception:
                pass
        if not closed_any:
            break


# ---------------------------
# AP Brands 메인에서 브랜드 상세 링크 수집
# ---------------------------
async def collect_brand_links(page):
    await page.goto(AP_BRANDS_MAIN, wait_until="domcontentloaded")
    await page.wait_for_timeout(800)

    # 메인에서 브랜드 상세 href들만 수집 (클릭 X)
    hrefs = await page.evaluate("""
    () => {
      const a = Array.from(document.querySelectorAll("a[href]"));
      const out = [];
      for (const el of a) {
        const href = el.getAttribute("href") || "";
        if (!href) continue;
        // 브랜드 상세만: /int/ko/brands/<slug>.html
        if (href.includes("/int/ko/brands/") && href.endsWith(".html") && !href.endsWith("brands.html")) {
          out.push(href);
        }
      }
      // 중복 제거
      return Array.from(new Set(out));
    }
    """)
    # 절대경로로 정규화
    links = [urljoin(AP_BRANDS_MAIN, h) for h in hrefs]
    return links


def slug_from_url(url: str) -> str:
    path = urlparse(url).path
    base = os.path.basename(path)
    slug = base.replace(".html", "").strip()
    return slug or "unknown"


# ---------------------------
# AP 브랜드 상세페이지에서 intro + 공식사이트 url 얻기
# ---------------------------
async def extract_ap_intro_and_official_url(page):
    # intro: 본문에서 "Visit and Follow Us" 위쪽을 최대한 주로 긁기
    body_text = await page.evaluate("() => document.body ? document.body.innerText : ''")
    body_text = body_text.strip()

    # 공식 사이트 버튼 href
    official = None
    btn = page.locator("a:has-text('브랜드 사이트 방문하기')")
    if await btn.count() == 0:
        # title 속성에 있는 경우
        btn = page.locator("a[title*='브랜드 사이트 방문하기']")
    if await btn.count() > 0:
        try:
            official = await btn.first.get_attribute("href")
        except Exception:
            official = None

    if official:
        official = official.strip()
        # 상대경로면 AP 도메인 기준으로 합치기
        if official.startswith("/"):
            official = urljoin(AP_BRANDS_MAIN, official)

    # intro는 버튼/푸터 등 제거한 버전으로
    intro = clean_text(body_text)

    return intro, official


# ---------------------------
# 공식 사이트에서 "브랜드 스토리"로 이동 시도
# ---------------------------
async def click_brand_story(page, brand_hint: str):
    # 1) 페이지 내에 '브랜드 스토리'가 바로 있으면 클릭
    direct = page.locator("a:has-text('브랜드 스토리'), button:has-text('브랜드 스토리')")
    if await direct.count() > 0:
        try:
            await direct.first.click(timeout=5000)
            await page.wait_for_load_state("domcontentloaded")
            return True
        except Exception:
            try:
                h = await direct.first.element_handle()
                if h:
                    await page.evaluate("(el)=>el.click()", h)
                    await page.wait_for_load_state("domcontentloaded")
                    return True
            except Exception:
                pass

    # 2) ABOUT {브랜드} or ABOUT 메뉴 열고 -> 브랜드 스토리
    about_candidates = [
        f"ABOUT {brand_hint}",
        "ABOUT",
        f"About {brand_hint}",
        "About",
    ]
    for txt in about_candidates:
        loc = page.get_by_text(txt, exact=False)
        try:
            if await loc.count() > 0 and await loc.first.is_visible():
                try:
                    await loc.first.click(timeout=2500)
                except Exception:
                    h = await loc.first.element_handle()
                    if h:
                        await page.evaluate("(el)=>el.click()", h)
                await page.wait_for_timeout(600)
                # 열린 후 브랜드 스토리 찾기
                story = page.get_by_text("브랜드 스토리", exact=False)
                if await story.count() > 0:
                    try:
                        await story.first.click(timeout=4000)
                    except Exception:
                        h = await story.first.element_handle()
                        if h:
                            await page.evaluate("(el)=>el.click()", h)
                    await page.wait_for_load_state("domcontentloaded")
                    return True
        except Exception:
            pass

    # 3) '브랜드' 메뉴 클릭 -> 브랜드 스토리
    brand_menu = page.get_by_text("브랜드", exact=False)
    try:
        if await brand_menu.count() > 0 and await brand_menu.first.is_visible():
            try:
                await brand_menu.first.click(timeout=2500)
            except Exception:
                h = await brand_menu.first.element_handle()
                if h:
                    await page.evaluate("(el)=>el.click()", h)
            await page.wait_for_timeout(600)
            story = page.get_by_text("브랜드 스토리", exact=False)
            if await story.count() > 0:
                try:
                    await story.first.click(timeout=4000)
                except Exception:
                    h = await story.first.element_handle()
                    if h:
                        await page.evaluate("(el)=>el.click()", h)
                await page.wait_for_load_state("domcontentloaded")
                return True
    except Exception:
        pass

    # 4) 최후: 페이지 전체에서 'brand story' (영문) 링크
    en_story = page.locator("a:has-text('Brand Story'), button:has-text('Brand Story')")
    if await en_story.count() > 0:
        try:
            await en_story.first.click(timeout=4000)
            await page.wait_for_load_state("domcontentloaded")
            return True
        except Exception:
            pass

    return False


# ---------------------------
# 공식 사이트에서 스토리 텍스트 추출
# ---------------------------
async def extract_official_story_text(page):
    # DOMContentLoaded만 믿고 너무 빨리 긁지 않게 약간 대기
    await page.wait_for_timeout(800)
    await try_close_popups(page)

    raw = await page.evaluate("() => document.body ? document.body.innerText : ''")
    return clean_text(raw)


# ---------------------------
# 메인
# ---------------------------
async def main(headless=False):
    os.makedirs(OUT_DIR, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context()
        page = await context.new_page()

        links = await collect_brand_links(page)
        print(f"총 브랜드 수: {len(links)}")

        for idx, brand_url in enumerate(links, start=1):
            slug = slug_from_url(brand_url)
            print(f"[{idx}/{len(links)}] 브랜드 처리 시작: {slug}")

            # (중요) 매 브랜드마다 AP brands 메인 거쳐서 들어가기
            await page.goto(AP_BRANDS_MAIN, wait_until="domcontentloaded")
            await page.wait_for_timeout(700)

            # 브랜드 상세로 이동
            await page.goto(brand_url, wait_until="domcontentloaded")
            await page.wait_for_timeout(900)

            ap_intro, official_url = await extract_ap_intro_and_official_url(page)

            story_text = ""
            official_final = official_url or ""

            if official_url:
                # 공식 사이트는 새 탭으로 (AP 페이지는 유지)
                official_page = await context.new_page()
                try:
                    await official_page.goto(official_url, wait_until="domcontentloaded", timeout=45000)
                    await try_close_popups(official_page)

                    # 브랜드 힌트(슬러그)로 ABOUT {브랜드} 찾는 데 도움
                    brand_hint = slug.upper()

                    moved = await click_brand_story(official_page, brand_hint=brand_hint)
                    await try_close_popups(official_page)

                    if moved:
                        story_text = await extract_official_story_text(official_page)
                    else:
                        # 스토리 못 찾으면: 일단 메인이라도 긁되, 필요 시 후처리
                        story_text = await extract_official_story_text(official_page)

                except PWTimeoutError:
                    story_text = "(공식 사이트 진입 타임아웃)"
                except Exception as e:
                    story_text = f"(공식 사이트 처리 중 오류: {e})"
                finally:
                    try:
                        await official_page.close()
                    except Exception:
                        pass

            # 저장
            out_path = os.path.join(OUT_DIR, f"{slug}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("=== [AP Group Intro] ===\n")
                f.write(ap_intro.strip() + "\n\n")
                f.write("=== [Official Site URL] ===\n")
                f.write((official_final or "(없음)") + "\n\n")
                f.write("=== [Brand Story / Official Site Text] ===\n")
                f.write(story_text.strip() + "\n")

            print(f"저장 완료: {out_path}")

        await browser.close()


if __name__ == "__main__":
    # headless=False 로 두면 브라우저 보면서 디버깅 가능
    asyncio.run(main(headless=False))
