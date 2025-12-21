import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

OUT = Path("laneige_identity.txt")

async def force_scroll(page, rounds=20, wait=1200):
    for _ in range(rounds):
        await page.evaluate("window.scrollBy(0, window.innerHeight)")
        await page.wait_for_timeout(wait)

async def collect_text(page):
    texts = []
    for el in await page.locator("h1, h2, h3, p").all():
        try:
            t = (await el.inner_text()).strip()
            if len(t) > 25:
                texts.append(t)
        except:
            pass
    return "\n\n".join(texts)

async def goto_laneige_from_hub(page):
    # brands 허브에서 라네즈 브랜드 상세 URL 찾기
    laneige_url = await page.evaluate("""
    () => {
        const links = Array.from(document.querySelectorAll("a"));
        for (const a of links) {
            if (a.href && a.href.includes("/brands/laneige.html")) {
                return a.href;
            }
        }
        return null;
    }
    """)
    if not laneige_url:
        raise RuntimeError("LANEIGE 브랜드 URL을 찾지 못했습니다.")

    await page.goto(laneige_url, timeout=0)

async def click_brand_site_button(page):
    # 아래로 내려야 버튼이 렌더링됨
    await force_scroll(page, rounds=12)

    button = page.locator("a:has-text('브랜드 사이트 방문하기')")
    await button.first.click()

    # 새 탭 열리면 그 탭으로 이동
    await page.wait_for_timeout(3000)
    pages = page.context.pages
    if len(pages) > 1:
        return pages[-1]
    return page

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        # 1️⃣ 아모레퍼시픽 브랜드 허브
        await page.goto(
            "https://www.apgroup.com/int/ko/brands/brands.html",
            timeout=0
        )
        await page.wait_for_timeout(3000)

        # 2️⃣ 라네즈 브랜드 상세 페이지 진입
        await goto_laneige_from_hub(page)
        await page.wait_for_timeout(3000)

        # 3️⃣ 브랜드 사이트 방문하기 버튼 클릭
        page = await click_brand_site_button(page)

        # 4️⃣ 브랜드 스토리 페이지로 이동
        await page.goto(
            "https://www.laneige.com/kr/ko/brand/open-to-wonder/",
            timeout=0
        )

        # 5️⃣ 스크롤해야 콘텐츠 로딩됨
        await force_scroll(page, rounds=35)

        # 6️⃣ 텍스트 수집
        text = await collect_text(page)
        OUT.write_text(text, encoding="utf-8")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())