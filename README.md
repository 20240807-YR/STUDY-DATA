# 아모레퍼시픽 Agent 10: 고객 감정 기반 CRM 메시지 자동 생성
https://amorepacific-ai.notion.site

🔗 프로젝트 주제: 고객 감정(리뷰/문의/불만/칭찬)을 분석해
상황·감정에 맞는 맞춤형 CRM 메시지를 자동 생성하는 Agent 10 설계

## 1. 주차별 상세 계획 (Detailed Weekly Plan)

### 1️⃣ 주차: ‘무엇을’ 풀 것인가? (도메인·유즈케이스 확정)

**핵심 목표**:
아모레퍼시픽 맥락에서 Agent 10이 다룰 고객 상황·감정·채널을 정리하고, “어떤 입력 → 어떤 출력”을 만들지 정의.

* Action Items:
	1.	핵심 유즈케이스 정의
	•	예:
	•	“배송 지연 항의 고객에게 사과 + 보상 안내 메시지 자동 생성”
	•	“피부 트러블 불만 리뷰에 공감·가이드 메시지 생성”
	•	“VIP 고객의 긍정 리뷰에 감사 + 추가 제안 메시지 생성”
	2.	입·출력 스펙 설계
	•	입력: 텍스트(리뷰/문의/콜센터 로그 요약), 메타 정보(채널, 제품군, 고객 등급 등)
	•	출력: 톤·길이·목적(사과/설명/제안/안내)이 명시된 CRM 메시지 템플릿
	3.	데이터 소스 및 스키마 정리
	•	감성 라벨용 텍스트 데이터(리뷰, SNS, 테스트용 감성 코퍼스)
	•	CRM 메시지 예시(실제/가상 템플릿) 수집·정리
	4.	Git / 폴더 구조 세팅
	•	data/, eda/, model/, agent/, notebooks/, docs/ 등 기본 구조 확정

⸻

### 2️⃣ 주차: ‘왜’ 이 데이터인가? (EDA 및 감성·상황 기준 설계)

**핵심 목표**:
Agent 10이 사용할 감성 레이블·상황 레이블 체계를 만들고, 텍스트 데이터가 이 기준과 잘 맞는지 EDA로 확인.

* Action Items:
	1.	감성·상황 라벨 체계 설계
	•	감성 예: 부정 / 중립 / 긍정 + 강도(매우, 보통 등)
	•	상황 예: 배송 / 제품 품질 / 피부 트러블 / 상담 경험 / 가격 / 프로모션 등
	•	“라벨 정의 문서” 작성 (각 라벨 예시 문장 포함)
	2.	EDA 코드 작성 및 공유
	•	텍스트 길이 분포, 감성별 비율, 상황별 비율, 키워드 빈도, n-gram, 대표 문장 추출
	•	부정 문장 상위 키워드, 긍정 문장 상위 키워드 분석 (지금 만든 sentiment notebook 활용)
	3.	라벨링 전략 정리
	•	완전 수동 라벨 / 규칙 기반 프리라벨 / 기존 감성모델 활용 등 전략 합의
	•	노이즈 허용 범위, 최소 데이터 개수, 학습·검증 분할 기준
	4.	“왜 이 데이터인가?” 논리 준비
	•	아모레퍼시픽 맥락에서 이 데이터가
	•	어떤 고객 경험을 대표하는지
	•	어떤 문제(이탈, 불만, CS 비용 등)를 줄이는 데 도움을 주는지 서술

⸻

### 3️⃣ 주차: ‘어떻게’ 성능을 높일 것인가? (메시지 생성 품질 개선)

**핵심 목표**:
감성·상황 분류 모델 + 메시지 생성 로직(프롬프트 or 파인튜닝)을 반복 개선해, **“사람이 쓴 것 같은 CRM 메시지”**에 최대한 근접.

* Action Items:
	1.	베이스라인 파이프라인 구축
	•	Step 1: 텍스트 → 감성/상황 라벨 예측 (기본 분류 모델 or 기존 감성모델 활용)
	•	Step 2: 라벨·메타정보를 입력으로 하는 메시지 생성(LLM 프롬프트 베이스라인)
	2.	개선 실험 (브랜치 단위 작업)
	•	감성·상황 분류 성능 개선
	•	전처리 개선, stopwords, 도메인 특화 단어 사전 추가
	•	간단한 ML 모델(Logistic, XGBoost) vs 사전학습 모델 비교
	•	메시지 생성 품질 개선
	•	프롬프트 설계 패턴 실험 (역할, 제약조건, 예시 few-shot)
	•	톤/채널별 스타일 컨트롤(카카오톡, 이메일, 앱푸시 등)
	3.	평가 기준 설계 및 검증
	•	정량: 감성·상황 분류 정확도, F1 등
	•	정성: 사람이 평가하는 “공감도/명확성/브랜드 톤 적합도” 체크리스트 설계
	•	실패 케이스(부적절한 톤, 정보 누락, 책임 회피처럼 느껴지는 표현 등) 모아서 개선

⸻

### 4️⃣ 주차: ‘어떻게’ 포장할 것인가? (Agent 10 제안서 및 데모 스토리)

**핵심 목표**:
Agent 10을 “실제 마케팅/CRM 팀이 바로 써보고 싶어지는 도구”로 보이도록 스토리와 결과물을 정리.

* Action Items:
	1.	Agent 10 컨셉 정리
	•	한 문장 정의:
	•	예) “고객 감정과 상황을 읽고, 아모레퍼시픽 톤에 맞는 CRM 메시지를 자동으로 제안하는 내부용 AI 어시스턴트”
	•	대표 시나리오 3~5개 작성
	•	“배송 지연 항의 → 사과 + 보상 안내 메시지 초안 생성”
	•	“트러블 후기 → 공감 + 사용 중지 안내 + 대체 제품 추천 메시지 생성”
	•	“장기 VIP 고객 리뷰 → 감사 + 전용 혜택 소개 메시지 생성”
	2.	제안서 작성
	•	문제 정의: 왜 ‘고객 감정 기반 CRM’이 중요한지
	•	데이터 & EDA: 어떤 데이터로 어떤 인사이트를 얻었는지
	•	모델 구조: 감성/상황 분류 → 메시지 생성 흐름 다이어그램
	•	성능 및 사례: 전/후 비교 예시, 실패 사례와 개선 과정
	3.	데모/플로우 정리
	•	입력 예시 텍스트 → Agent 10의 응답 CRM 메시지 예시 스크린샷/텍스트
	•	“사람이 수정만 하면 곧바로 발송 가능한 수준”을 강조
	4.	코드 & 리포 정리
	•	main 기준으로 노트북·모델·유틸리티 구조 정리
	•	주석·함수 이름·README 정리
	•	최종 제안서 및 코드 제출

⸻

## 📅 12월 2–3일 작업 요약: 네이버 뷰티 검색 기록 분석 / 쇼핑 데이터 수집 / 인구·GRDP 분석 / 감성 분석 기초 모델 준비

### 12_02_sentiment.ipynb — 감성 분석 기초 모델 구축(고객 CRM 메시지 생성용)

* **데이터 로드 및 정리** 
    * `테스트용.csv` 로드(latin-1 인코딩)하였습니다. 
    * 감성 레이블(0 부정 / 4 긍정 / 2 중립) 구조 이해하였습니다.
    * 불필요한 `flag` 컬럼 `drop → df_model = [label, text]` 구성하였습니다. 

* **텍스트 전처리** 
    * `nltk stopwords` 다운로드하였습니다.
    * URL 제거, @유저 제거, 알파벳·공백 제외 정리, custom stopwords(im, ive, dont, rt, amp …) 추가하였습니다. 

* **부정(label=0) 텍스트 단어 분석** 
    * tqdm 기반 전체 토큰 분석하여 Counter로 단어 `frequency` 계산하였습니다. 
    * 최종 전처리 함수 확장 및 token 확인`(clean_text("RT …"))`하였습니다. 

### 12_03_네이버뷰티검색기록.ipynb — 네이버 뷰티 검색 기록 분석

* **데이터 로드 및 전처리**
* * **데이터 로드 및 정리** 
    * `네이버뷰티검색기록.csv` 로드 후 헤더 시작 위치를 탐색하기 위해 날짜 정규식`(\d{4}-\d{2}-\d{2})`으로 실제 데이터 시작행(start_idx) 자동 탐지하였습니다. 
    * `skiprows=6` 적용하여 본 데이터 프레임 재로드하였습니다.  
    * 날짜 컬럼 → datetime 변환, 스킨케어·트러블케어·선케어·클렌징·메이크업 등 주요 카테고리만 추출해 clean_df 생성하였습니다.
    * index를 날짜로 설정하고 숫자형으로 정리하였습니다. 

* **검색량 추세 분석**
	* AppleSDGothic 폰트 적용 후 카테고리별 검색량 추세 그래프 시각화하였습니다.

* **증감률 분석**
	* 일별 증감률(pct_change), 전년 대비 증가율(yoy), 월별 평균 증가율(MoM) 계산하였습니다.
	* 0만 포함하는 행 제거 후 재정제.

* **시즌성 분석**
	* 카테고리별 최고 관심도 시점(peak) / 최저 관심도 시점(low) 산출하였습니다.

* **상관 분석**
	* 카테고리 간 상관행렬 계산 후 `heatmap` 시각화하였습니다.

* **요일 패턴 분석**
	* weekday 컬럼 생성하여 요일별 평균 검색량 집계 및 bar plot 시각화하였습니다.

⸻

### 12_03_쇼핑.ipynb — 네이버 쇼핑인사이트 스킨케어 데이터 수집(크롤링)

**네이버 쇼핑 Insight 자동 다운로드**
	* `selenium + ChromeDriverManager` 사용해 브라우저 자동 실행하였습니다.
	* 카테고리를 1차: 화장품/미용, 2차: 스킨케어 선택하여 쇼핑 검색량 CSV 자동 다운로드하였습니다.

⸻

### 12_03_인구데이터성별.ipynb — 20~34세 여성 인구 분석

* **데이터 로드 및 기본 전처리**
	* 인구데이터성별.csv` 로드하여 컬럼명(region, sex, age, pop) 정리하였습니다.
	* “연령별”에서 “20~34세 여성”만 추출하여 분석하였습니다.

* **주요 처리**
	* 20~34세 여성 전체 인구 합계 계산하였습니다.
	* 지역별 여성 20~34세 인구 Top10 도출하여 반올림 처리하였습니다.
	* 수도권(서울·경기·인천) 집중도 계산하였습니다.
	* 지역 전체 인구 대비 20–34세 여성 비중 산출하였습니다.
	* 연령대(20–24 / 25–29 / 30–34) 구간별 지역 분포 테이블 생성하였습니다.
	* bar chart로 Top10 시각화하였습니다.

⸻

### 12_03_인구통계.ipynb — 장기 시계열 기반 여성2034 인구 분석

* **데이터 로드 및 정리**
	* `인구통계.csv` 로드, 컬럼 재정의(region, age, sex, time, population)하였습니다.
	* `latest = df["time"].max()` 로 최신 시점 추출하였습니다.

* **주요 분석** 
  	* 최신 시점 기준 20~34세 여성 데이터 추출하였습니다.
	* 지역별 인구 합산 후 Top10 도출하였습니다.
	* 전체 데이터 개수 / 여성 데이터 개수 / 최신 시점 데이터 개수 등 기본 검증 출력하였습니다.

⸻

### 12_03_GRDP.ipynb — GRDP·소득·소비 분석 및 클러스터링

* **데이터 전처리**
	* `GRDP시도별.csv` 로드 → GRDP, 소득, 민간소비 컬럼 정리하였습니다.
	* 수치형 변환하였습니다.

* **데이터 전처리**
  	* GRDP 상위 TOP10 시각화(bar)하였습니다.
	* 민간소비 / 개인소득 비율(consumption_ratio) 계산하였습니다.
	* 소비 성향 상위 Top10 시각화하였습니다.
	* 소비 성향을 4개 클러스터로 분류(보수적/평균/적극적/초고소비)하였습니다.

* Beauty Market Potential Score 기초 구상하였습니다
	•	Potential = (2034 여성 인구 비중 × 0.5) + (소비성향 × 0.3) + (GRDP × 0.2)

⸻

## 📅 12월 4일일 작업 요약: 올리브영 세일·리뷰·모델링 전처리

### 12_04_올영.ipynb — 올리브영 세일·랭킹·오특·핫딜 데이터 크롤링
이날은 올리브영 세일·핫딜 상품 데이터를 대량으로 크롤링하고, 가격·할인율·리뷰 정보를 합쳐 가성비/할인 전략/브랜드 가치를 평가하는 모델을 만드는 작업을 진행했습니다.

* **올리브영 세일·랭킹·오특·핫딜 데이터 크롤링** 
    * `Selenium 기반 크롤러`를 구현해 메인/탭/전용 URL에서 상품 리스트를 수집했습니다.
    * `crawl_oliveyoung_best`, `crawl_oliveyoung_new`, `crawl_oliveyoung_tab(“오특”, “랭킹”, “세일”, “기획전”)`, `crawl_hotdeal`, `crawl_sale_all` 등 함수 작성.
    * 각 상품에 대해 이름, 현재가, 정가, 리뷰 수치(텍스트 숫자), 평점 등을 수집해 `올영_베스트.csv`, `올영_핫딜.csv`, `올영_세일.csv` 등 CSV로 저장했습니다.

### 12_04_올영예측순위.ipynb — 가격·할인율 기반 랭킹/가성비 예측 모델링

* **가격·할인율 기반 랭킹/가성비 예측 모델링**
    * 피처: `price_num`, `orig_price_num`, `discount_rate` / 타깃: `rank` 로 두고 `train/test split` 후 여러 회귀 모델을 테스트했습니다.
    * `Linear Regression`, `RandomForestRegressor`, `GradientBoostingRegressor`, `XGBRegressor` 등으로 `MAE·R²` 비교하였습니다.
    * 상위 10% 상품을 1로 두는 이진 타깃 is_top 를 만들어 `Logistic Regression`으로 Top 상품 분류(`Accurac`y, `ROC-AUC`, 계수 해석)도 진행했습니다.
    * 학습된 XGBoost 모델 출력을 이용해
예측 순위 → `pred_rank`
인기도 점수 → `popularity_score = 1 / pred_rank`
가성비 점수 → `value_score = popularity_score / price_num`를 정의하고, 가성비 TOP 20/30 상품 리스트를 추출했습니다.
	* 특정 상품에 대해 할인율을 10~50%로 바꿔가며 예측 순위 변화를 시뮬레이션하는 `simulate_discount_effect()` 함수로 “할인율 변화에 따른 예상 순위 개선”을 계산했습니다.

⸻

### 12_04_올영리뷰.ipynb — 리뷰 메타데이터 수집 + 리뷰 텍스트 수집 및 감성 분석

* **리뷰 메타데이터 수집 (리뷰 개수·평점)** 
    * 세일 상품 URL에서 정규식으로 `goodsNo`를 추출해 상품 고유번호 컬럼을 만들었습니다.
    * `getGoodsEvalList.do API에 goodsNo`를 넘겨 각 상품의 리뷰 총 개수(totalCount)와 평균 평점(avgScore) 를 받아와 `review_count, review_score` 컬럼으로 저장하고 `올영_세일_리뷰추가.csv` 를 생성했습니다.

* **리뷰 텍스트 수집 및 간단 감성 분석** 
    * API 응답에 텍스트가 비어 있는 문제를 확인하고, 여러 방식으로 리뷰 텍스트를 수집하는 시도를 했습니다.
    * `getGoodsReview.do POST, Selenium`으로 상세페이지 리뷰 탭 클릭 후 `DOM 파싱`, `JS fetch`를 페이지 내부에서 실행, `__PRELOADED_STATE__` 같은 `preloaded JSO`N 탐색, Network 로그에서 리뷰 API 후보 URL 자동 수집, 모바일(m.oliveyoung) 리뷰 API(/review/api/v2/reviews) 호출 등.
    * 최종적으로 리뷰 텍스트 + 평점 + 작성일을 모아 `올영_세일_리뷰텍스트.csv` 로 저장했습니다.
    * 리뷰 텍스트에 대해:
	특수문자 제거·공백 정리로 `content_clean` 생성
	공백 기준 토큰화 후 길이 2글자 미만 제거 → `tokens` 리스트 생성
	전체 토큰 빈도로 TOP 키워드 50개 확인하여 간단한 사전 기반 감성 분석을 구현했습니다.
	* 긍정 단어 리스트(‘좋’, ‘만족’, ‘추천’, ‘촉촉’ 등), 부정 단어 리스트(‘별로’, ‘실망’, ‘건조’, ‘자극’ 등) 정의하여 문장에 포함된 단어 수로 `sentiment_score` 계산하였습니다. `score>0: pos`, `<0: neg`, `0: neu로 sentiment_label` 부여하였습니다.
 	* 리뷰 DF와 상품 DF를 `goodsNo`로 `merge`해 브랜드별 리뷰 수·평균평점·긍정/부정 비율을 집계한
`brand_review_stats`, `상품별 리뷰 수·평점·평균 감성 점수 집계(prod_review_stats)`를 생성했습니다.

⸻

### 12_04_올영하이브리드.ipynb — 리뷰·감성 지표를 포함한 고도화 모델 및 할인 전략 지표

* **리뷰·감성 지표를 포함한 고도화 모델 및 할인 전략 지표** 
    * 상품 DF에 리뷰 통계를 병합해 `n_reviews`, `avg_review_score`, `avg_sentiment` 
	피처를 추가하고, 리뷰 50개 이상 보유 상품만 모은 `df_model_hr`를 구성했습니다.
    * 피처: `price_num`, `orig_price_num`, `discount_rate`, `n_reviews`, 
	`avg_review_score`, `avg_sentiment`
	* 타깃: `rank` 로 `XGBRegressor`를 다시 학습해 리뷰 정보가 순위 예측에 주는 영향
	(feature importance)까지 확인했습니다.
	* 각 상품에 대해 할인율을 +10% 올렸을 때 예측 순위가 얼마나 개선되는지 계산하는 
	`discount_sensitivity` 변수를 만들고, 0~1 사이로 정규화한 `popularity_boost_score`
	를 정의했습니다.
	* `기존 가성비 점수(value_score)`와 결합해 `hybrid_score = value_score * (1 + 
	popularity_boost_score)`를 만들고, 할인했을 때 인기 상승 여지가 크면서도 현재 가성비가 좋
	은 상품을 TOP 30 리스트로 추려냈습니다.
	* 브랜드별 평균 `hybrid_score`를 계산해 **“가격/할인/리뷰/할인 민감도까지 합친 종합 가치가 높
	은 브랜드 TOP 30”**을 가로 막대 그래프로 시각화했습니다.
 
⸻

### 12_05_초기.ipynb — 데이터 컬럼 점검·시장 페르소나 스코어링·하이브리드 추천 엔진 v0
* **columns 확인** 
    * `올영_세일.csv`, `네이버뷰티검색기록.csv`, `올영_베스트.csv`, `올영_세일_리뷰추가.csv`, `올영_세일파이널.csv`, `올영_핫딜.csv`, `인구데이터성별.csv`, `인구통계.csv`, `GRDP시도별.csv`를 순차적으로 `pd.read_csv()` 후 `df.columns` 출력해서 주요 컬럼 구조를 확인했습니다.

* **초기 페르소나 점수 계산 코드(v0)** 
    * 여러 CSV(베스트, 세일, 세일_리뷰추가, 핫딜, 인구데이터성별)를 불러와 `clean_number() /
	to_num()` 함수로 가격·평점·리뷰 수를 숫자형으로 정리했습니다.
    * `Clean Ingredient` / `Emotional` / `Value / `Premium / `Functional` / `Routine` / `Simplicity` / `Trend Responsiveness` 7개 스코어를 계산해 `persona_scores` 딕셔너리와 `persona_df`로 정리, `barh` 그래프로 시각화했습니다.
    * 주석으로 각 스코어(예: `Value Sensitivity Score ≈ 0.29`, `Premium Orientation
	Score ≈ 0.5`, `Trend Responsiveness Score`가 매우 낮은 이유 등) 해석 메모 작성했습니다.

* **페르소나 스코어 정규화 및 해석**
	* `persona_df`에서 `score` 컬럼을 `Min–Max` 정규화해 `normalized_score` 추가.
 	* 정규화 결과를 기반으로 
	`Premium = 1.0`, `Functional = 1.0`, `Routine Simplicity ≈ 0.67`, `Value ≈ 0.59`,`Trend ≈ 0.05`, `Clean & Emotional = 0.0` 등 시장 `baseline` 성향을 텍스트로 정리했습니다.
	
* **XGBoost 기반 하이브리드 스코어 계산**
	* `올영_세일.csv`에서 `price_num`, `orig_price_num`, `discount_rate`, `rank` 생성 후 `XGBRegressor(300 trees, learning_rate=0.05, max_depth=4 등)`로 `rank` 예측 모델 학습했습니다.
 	* 전체 상품에 대해 `pred_rank`, `popularity_score = 1` / `(pred_rank + 1),
value_score = popularity_score / (price_num + 1)` 계산했습니다.
	* `compute_discount_sensitivity()` 함수로 할인율 +10% 가정 시 순위 개선량을 시뮬레이션 → `discount_sensitivity → popularity_boost_score` 정규화 →
최종 `hybrid_score = value_score * (1 + popularity_boost_score)` 계산했습니다.
 	* 상품명에서 첫 단어로 `brand`를 추출하고 브랜드별 평균 `hybrid_score` 집계 후 TOP 30 가로 막대 시각화했습니다.

* **시장 baseline 기반 persona_df 재계산 + persona 엔진 v1**
	* `df`에 `hybrid_score`, `discount_rate`, `brand`, `popularity_boost_score`가 존재하는 상태를 가정하고, 없을 경우를 대비한 보정 코드 포함했습니다.
 	* 외부 데이터(올영_베스트, 올영_세일_리뷰추가, 올영_핫딜, 인구데이터성별)를 다시 불러와
`Clean` / `Emotional` / `Value` / `Premium` / `Functional` / `Routine` `Simplicity` / `Trend` 스코어를 재계산하고 `Min–Max` 정규화 → `persona_df`, `persona_scores.`
	* `premium_brands` = ["설화수", "헤라"], `value_brands` = ["라네즈", "이니스프리"],`clean_brands` = ["닥터지", "일리윤", "라로슈포제"], f`unctional_brands` = ["닥터지", "메디힐", "피지오겔"]를 정의했습니다.

* **persona_affinity + final_reco_score 기반 추천 점수**
	*` compute_persona_affinity() / compute_affinity()` 함수에서
브랜드군, `discount_rate`, `popularity_boost_scor`e와 `persona_scores`를 조합해 `persona_affinity` 계산했습니다.
 	* `hybrid_score_norm`, `persona_affinity_norm`으로 정규화 후
`final_reco_score = 0.6 * hybrid_score_norm + 0.4 * persona_affinity_norm` 정의했습니다.
	* `final_reco_score` 기준 TOP 20 및 TOP 3 상품을 정렬해 출력하고,
`format_item()`으로 “이름·가격·할인율·URL” 형식의 텍스트 블록 생성했습니다.

* **persona_affinity + final_reco_score 기반 추천 점수**
	* `microsoft/Phi-3-mini-4k-instruct`를 `transformers`로 로드하고,
`llm_generate(prompt)` 래퍼 함수를 정의해 추천 결과와 페르소나 설명을 LLM 프롬프트로 넘길 준비했습니다.

⸻

### 12_06_아모레.ipynb — 아모레퍼시픽(라네즈) 베스트셀러 크롤링
* **라네즈 베스트셀러 페이지 크롤링** 
    * `requests + BeautifulSoup`으로 아모레몰 라네즈 베스트셀러 페이지 HTML을 요청했습니다.
    * `div.productCard.shop-productCard.typeSmall` 요소를 기준으로 상품 카드 리스트를 수집했습니다.
    * 각 상품에서 `.name -> 상품명`, `.price -> 가격`, `.discount -> 할인율`, `.review -> 평점`을 추출하여 `rows` 리스트에 저장했습니다. 

* **CSV 저장** 
    * 아모레몰에 있는 전체 브랜드의 csv 파일을 생성했습니다.
    * 헤더는 ["상품명", "가격", "할인율", "평점"]으로 구성했습니다.
    * `UTF-8-SIG`로 저장하여 엑셀·한글에서 깨지지 않도록 처리했습니다.

