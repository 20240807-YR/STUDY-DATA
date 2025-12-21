import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

BRANDS_URL = "https://www.apgroup.com/int/ko/brands/brands.html"
OUT_DIR = Path("brand_identity_txt")
OUT_DIR.mkdir(exist_ok=True)

async def slow_scroll(page, times=20, delay=400):
    for _ in range(times):
        await page.mouse.wheel(0, 1200)
        await page.wait_for_timeout(delay)

async def extract_text(page):
    return await page.evaluate("""
        () => Array.from(document.querySelectorAll("h1,h2,h3,p,span"))
            .map(e => e.innerText.trim())
            .filter(t => t.length > 25)
            .join("\\n")
    """)

async def enter_brand_story(page, brand_name):
    # 설화수 케이스
    about = page.locator(f"text=ABOUT {brand_name}")
    if await about.count() > 0:
        await about.first.click()
        await page.wait_for_timeout(1500)

    # 라네즈 등 일반 케이스
    brand_menu = page.locator("text=브랜드")
    if await brand_menu.count() > 0:
        await brand_menu.first.click()
        await page.wait_for_timeout(1200)

    story = page.locator("text=브랜드 스토리")
    if await story.count() == 0:
        return False

    await story.first.click()
    await page.wait_for_timeout(2000)
    return True

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        # 브랜드 개수는 brands.html에서만 읽는다
        await page.goto(BRANDS_URL)
        await page.wait_for_timeout(2000)
        brand_count = await page.locator("a[href*='/brands/']").count()
        print(f"총 브랜드 수: {brand_count}")

        for i in range(brand_count):
            print(f"[{i+1}/{brand_count}] 브랜드 처리 시작")

            # 🔴 핵심: 매번 brands.html로 복귀
            await page.goto(BRANDS_URL)
            await page.wait_for_timeout(2000)

            cards = page.locator("a[href*='/brands/']")
            card = cards.nth(i)
            brand_name = (await card.inner_text()).strip()
            if not brand_name:
                brand_name = f"brand_{i}"

            await card.click()
            await page.wait_for_timeout(2000)
            await slow_scroll(page, 10)

            # 브랜드 사이트 버튼
            site_btn = page.locator("text=브랜드 사이트 방문하기")

            if await site_btn.count() > 0:
                async with context.expect_page() as pinfo:
                    await site_btn.first.evaluate("(el) => el.click()")
                brand_page = await pinfo.value
                await brand_page.wait_for_load_state()
            else:
                brand_page = page

            entered = await enter_brand_story(brand_page, brand_name)
            if not entered:
                print("브랜드 스토리 없음 → 스킵")
                if brand_page != page:
                    await brand_page.close()
                continue

            await slow_scroll(brand_page, 25)
            text = await extract_text(brand_page)

            if text.strip():
                out = OUT_DIR / f"{brand_name}.txt"
                out.write_text(text, encoding="utf-8")
                print(f"저장 완료: {out}")

            if brand_page != page:
                await brand_page.close()

        await browser.close()

asyncio.run(main())