import pandas as pd
import numpy as np
import os  # 파일 존재 여부 확인
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

# --- 분석기 클래스 (기존과 동일) ---
class TextAnalyzer:
    def __init__(self):
        print(">>> [1/2] 형태소 분석기(Kiwi) 로딩 중...")
        try:
            self.kiwi = Kiwi(num_workers=1)
        except Exception:
            self.kiwi = Kiwi()
        
        print(">>> [2/2] AI 톤 분석 모델(Zero-Shot) 로딩 중...")
        self.classifier = pipeline(
            "zero-shot-classification", 
            model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
            device=-1 
        )
        print(">>> 초기화 완료! 분석을 시작합니다.\n")

    def analyze_style_pattern(self, text):
        tokens = self.kiwi.tokenize(text)
        metrics = {'adjectives': 0, 'verbs': 0, 'formal_ending': 0}
        
        for token in tokens:
            if token.tag.startswith('VA'): 
                metrics['adjectives'] += 1
            elif token.tag.startswith('VV'): 
                metrics['verbs'] += 1
            if token.tag.startswith('EF') and any(x in token.form for x in ['니다', '니까', '십시오']):
                metrics['formal_ending'] = 1
        return metrics

    def predict_tone(self, text, labels):
        target_text = text[:512] 
        result = self.classifier(target_text, labels, multi_label=False)
        return result['labels'][0], result['scores'][0]

    def generate_report(self, documents, tone_labels):
        def tokenizer(text):
            tokens = self.kiwi.tokenize(text)
            return [t.form for t in tokens if t.tag.startswith('NN')]

        vectorizer = TfidfVectorizer(tokenizer=tokenizer, min_df=1)
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = np.array(vectorizer.get_feature_names_out())

        results = []
        for idx, doc in enumerate(documents):
            feature_index = tfidf_matrix[idx, :].nonzero()[1]
            tfidf_scores = zip(feature_index, [tfidf_matrix[idx, x] for x in feature_index])
            sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
            top_keywords = [feature_names[i] for i, score in sorted_scores[:5]]

            style = self.analyze_style_pattern(doc)
            tone, score = self.predict_tone(doc, tone_labels)

            results.append({
                "구분": f"Part {idx+1}",
                "내용 요약": doc.strip()[:30].replace("\n", " ") + "...",
                "핵심 키워드": ", ".join(top_keywords),
                "AI 분석 톤": tone,
                "확신도": f"{score:.2f}",
                "형용사 수": style['adjectives'],
            })
        
        return pd.DataFrame(results)

# --- 실행부 (데이터 원문 100% 반영: 이니스프리) ---
if __name__ == "__main__":
    
    # ==========================================
    # 1. 설정: 브랜드 이름 변경
    # ==========================================
    current_brand_name = "이니스프리"

    # 2. 데이터 입력 (보내주신 텍스트 전체 포함)
    
    # Part 1: 브랜드 컨셉 (Nature-Powered & Be Free)
    full_text_part1 = """
    Effective, Nature-Powered Skincare
    Discovered from the Island
    무한한 자연의 에너지를 탐구해 건강한 아름다움을 전하는 고효능 자연주의 브랜드
    Effective, Nature-Powered
    Skincare Discovered from the Island
    무한한 자연의 힘을 담은 다채로운 식물과
    생명력 가득한 토양을 품은 대지,
    그리고 미지의 에너지로 가득한 청록의 바다
    Enjoy being you,
    Be free with INNISFREE
    이니스프리와 함께 겉으로 보이는 아름다움을 넘어
    당신의 모든 아름다운 찰나를 발견해보세요.
    """

    # Part 2: 헤리티지 및 핵심 원료 (Jeju Origin & Green Tea)
    full_text_part2 = """
    the Origin of INNISFREE
    천혜의 자연환경과 경이로운 생명력을 가진 제주.
    그곳의 흙과 빛, 물과 바람, 그리고 안개가 조화롭게 어우러진
    청록의 다원은 이니스프리의 시작점입니다.
    돌무더기와 가시덤불이 가득했던 황무지를
    손으로 일구어 비옥한 차밭으로
    개간한 창업주의 끈질긴 집념과 개척 정신은
    ‘건강한 피부를 위해 끊임없이 도전하고 연구’
    하는 오늘날 이니스프리의 근간이 되었고,
    이것의 산물이라고 할 수 있는 ‘그린티’는
    이니스프리의 정수가 되는 원료로 자리 잡았습니다.
    Farm to Face
    Beauty Green Tea™ INNISFREE
    녹차 나무 한 그루가 자라기까지 걸리는 시간 5년.
    약 20여년 동안 여러 실험 끝에 2만여 종의 녹차 중
    기후, 병충해를 비롯한 여러 악조건을 견디며
    최적의 효능을 발휘해낼 수 있는,
    전세계 유일무이한 ‘피부만을 위한 녹차 품종’을 탄생 시킬 수 있었습니다.
    """

    # Part 3: 지속가능성 및 사회공헌 (Sustainability & ESG)
    full_text_part3 = """
    with INNISFREE
    Better for Us and Earth
    이니스프리는 함께 살아가는 우리 모두와 지구를 위한 선택과 실천을 제안합니다.
    이니스프리와 함께할 수 있는 즐거운 실천들과
    그 동안 모두와 함께 만들어 온 아름다운 발자취를 지금 바로 확인해보세요.
    Sustainable Green Beauty
    이니스프리는 제품을 개발하는 초기 단계부터 제품의 내용물과
    패키지의 환경 영향력에 대해 고민하고 연구합니다.
    오늘보다 더 나은 내일을 위해 지구에 이로운 방향으로 제품을 개발하겠습니다.
    Vegan
    제품의 내용물에 불필요한 동물성 원료를 처방하지 않는
    비건 제품을 점차 확대해 나가겠습니다.
    Less Plastic
    분리배출이 용이한 단일 소재(PP메탈 프리 펌프 등)를
    사용한 제품을 늘려나가고,
    버려진 플라스틱과 유리를 재가공하여 만든 재생 원료를
    제품 용기와 캡에 적용하는 비율을 높여 나가겠습니다.
    또, 리필 가능한 용기를 사용하는 제품을 개발하여
    Less Plastic을 실천해 나가겠습니다.
    FSC(Forest Stewardship Council) 인증 Papers
    이니스프리는 플라스틱 비닐 소재의 포장재 대신
    FSC 인증을 받은 종이 포장재와 박스를 사용하여
    환경에 불필요하게 미치는 영향을 최소화 했습니다.
    배송박스를 받으신 후에는 종이류로
    분리 배출해 주세요!
    이니스프리 모음재단
    이니스프리는 브랜드의 시작점인 제주의 가치를 보전하기 위해
    2015년 이니스프리 모음재단을 설립했습니다. 오름 연구지원,
    숲 조성, 생물 다양성 보전 등 제주 다움의 근간인 자연환경보전 활동을
    전개하고 있습니다.
    또한 제주 그린어워드를 통해 제주 자연유산과 환경 보전을 위해
    활동하는 이들을 발굴·지원하고, 미래세대의 환경 감수성 향상을 위한
    교육을 운영하며 제주에 가치를 더하는 문화를 만들어가고 있습니다.
    """

    documents = [full_text_part1, full_text_part2, full_text_part3]
    target_tones = ["의학적/전문적인", "신뢰감/진정성있는", "도전적/혁신적인", "감성적/부드러운"]
    
    # 3. 분석 실행
    analyzer = TextAnalyzer()
    df_result = analyzer.generate_report(documents, target_tones)

    # 4. 브랜드 컬럼 추가 및 정리
    df_result['브랜드'] = current_brand_name
    cols = ['브랜드'] + [c for c in df_result.columns if c != '브랜드']
    df_result = df_result[cols]

    # 5. CSV 파일 누적 저장 (Append 모드)
    save_file_name = "total_brand_analysis.csv"
    
    if os.path.exists(save_file_name):
        df_result.to_csv(save_file_name, mode='a', index=False, header=False, encoding="utf-8-sig")
        print(f"\n✅ 기존 '{save_file_name}' 파일 아래에 '{current_brand_name}' 데이터를 추가했습니다.")
    else:
        df_result.to_csv(save_file_name, mode='w', index=False, header=True, encoding="utf-8-sig")
        print(f"\n📂 새 파일 '{save_file_name}'을 생성하고 '{current_brand_name}' 데이터를 저장했습니다.")

    # 터미널 확인용 출력
    print("="*80)
    print(df_result)
    print("="*80)