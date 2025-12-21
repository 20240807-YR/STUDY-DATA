import asyncio
import re
from pathlib import Path
from playwright.async_api import async_playwright

# 저장 경로 설정
OUT_DIR = Path("brand_identity_txt")
OUT_DIR.mkdir(exist_ok=True)

# 시작 URL
BASE_URL = "https://www.apgroup.com"
BRANDS_LIST_URL = "https://www.apgroup.com/int/ko/brands/brands.html"

async def slow_scroll(page, times=8, delay=500):
    """페이지 끝까지 스크롤하여 모든 텍스트 로딩"""
    for _ in range(times):
        await page.mouse.wheel(0, 1500)
        await page.wait_for_timeout(delay)

async def extract_text_content(page):
    """
    현재 페이지의 모든 가시적 텍스트 추출.
    Script, Style, Nav, Footer 등 노이즈 제거.
    """
    return await page.evaluate("""
        () => {
            // 본문 추정 영역(없으면 body 전체)
            const target = document.querySelector('main') || document.querySelector('#container') || document.body;
            const clones = target.cloneNode(true);
            
            // 노이즈 제거 (메뉴바, 푸터, 스크립트 등)
            const trashes = clones.querySelectorAll('script, style, noscript, header, footer, .gnb, .nav, .cookie');
            trashes.forEach(t => t.remove());
            
            const text = clones.innerText;
            // 공백 정리
            return text.split('\\n')
                .map(line => line.trim())
                .filter(line => line.length > 1) // 너무 짧은 줄(1글자) 제거
                .join('\\n');
        }
    """)

async def try_find_and_scrape_story(page):
    """
    다양한 키워드로 '브랜드 소개' 메뉴를 찾아 클릭하고 내용을 긁어옴.
    """
    # 검색할 메뉴 키워드 (영어/한국어 포함, 우선순위 순)
    keywords = [
        "Brand Story", "브랜드 스토리", 
        "About", "About Us", 
        "Philosophy", "철학", 
        "Concept", "브랜드 컨셉",
        "Our Story", "Heritage", 
        "World", "세계관",
        "Introduction", "소개"
    ]
    
    scraped_text = ""
    
    try:
        # 1. 햄버거 메뉴가 있다면 열어야 할 수도 있음 (모바일 대응)
        # PC버전에서는 보통 상단에 있으므로 바로 검색
        
        target_locator = None
        found_keyword = ""

        for kw in keywords:
            # 대소문자 구분 없이 텍스트 포함 요소 찾기
            loc = page.get_by_text(re.compile(kw, re.IGNORECASE))
            if await loc.count() > 0 and await loc.first.is_visible():
                target_locator = loc.first
                found_keyword = kw
                break
        
        if target_locator:
            print(f"   -> 메뉴 발견! ('{found_keyword}') 클릭 시도...")
            await target_locator.click()
            await page.wait_for_load_state('networkidle', timeout=5000)
            await slow_scroll(page, 5) # 이동 후 스크롤
            
            text = await extract_text_content(page)
            scraped_text = f"\n\n=== [Detailed Story Page: {found_keyword}] ===\n{text}"
        else:
            print("   -> '스토리' 관련 메뉴를 찾지 못했습니다. (메인 페이지만 저장)")
            
    except Exception as e:
        print(f"   -> 메뉴 탐색 중 에러 발생(무시): {e}")

    return scraped_text

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, args=["--start-maximized"]) # 브라우저 보면서 실행
        context = await browser.new_context(no_viewport=True)
        page = await context.new_page()

        print(">>> 1. 브랜드 목록 수집 시작")
        await page.goto(BRANDS_LIST_URL)
        await page.wait_for_load_state('networkidle')
        
        # URL 수집
        links = await page.locator("a[href*='/brands/']").all()
        collected_urls = set()
        for link in links:
            href = await link.get_attribute("href")
            if href and "brands.html" not in href and href.endswith(".html"):
                full_url = href if href.startswith("http") else BASE_URL + href
                collected_urls.add(full_url)

        sorted_urls = sorted(list(collected_urls))
        print(f"=== 총 {len(sorted_urls)}개 브랜드 URL 확보 ===")

        # [메인 루프]
        for idx, url in enumerate(sorted_urls):
            brand_name = url.split("/")[-1].replace(".html", "")
            print(f"\n[{idx+1}/{len(sorted_urls)}] {brand_name} 작업 시작")
            
            final_content = []

            try:
                # 1. AP Group 내부 페이지
                await page.goto(url)
                await page.wait_for_load_state('domcontentloaded')
                await slow_scroll(page, 2)
                
                ap_text = await extract_text_content(page)
                final_content.append(f"=== [AP Group Intro] ===\n{ap_text}")

                # 2. 공식 사이트 이동 버튼 찾기
                visit_btn = page.locator("a", has_text="브랜드 사이트 방문하기")
                
                if await visit_btn.count() > 0:
                    print("   -> 공식 사이트 진입...")
                    async with context.expect_page() as new_page_info:
                        await visit_btn.first.click()
                    
                    new_page = await new_page_info.value
                    await new_page.wait_for_load_state('domcontentloaded')
                    
                    # 팝업 닫기 시도 (간단하게)
                    try:
                        await new_page.locator(".btn_close, .close_layer, .modal .close").click(timeout=1000)
                    except:
                        pass
                    
                    # [중요 1] 진입하자마자 메인 페이지(Landing Page) 긁기
                    print("   -> 메인 페이지 내용 수집 중...")
                    await slow_scroll(new_page, 5)
                    main_text = await extract_text_content(new_page)
                    final_content.append(f"\n\n=== [Official Site Main Page] ===\n{main_text}")

                    # [중요 2] 스토리 메뉴 찾아서 추가로 긁기
                    story_text = await try_find_and_scrape_story(new_page)
                    if story_text:
                        final_content.append(story_text)
                    
                    await new_page.close()
                else:
                    print("   -> 공식 사이트 버튼 없음")

                # 파일 저장
                full_text = "\n".join(final_content)
                out_path = OUT_DIR / f"{brand_name}.txt"
                out_path.write_text(full_text, encoding="utf-8")
                print(f"   -> 저장 완료 ({len(full_text)}자)")

            except Exception as e:
                print(f"   -> ⚠️ 에러 발생: {e}")
                continue

        await browser.close()
        print("\n모든 작업 완료!")

if __name__ == "__main__":
    asyncio.run(main())