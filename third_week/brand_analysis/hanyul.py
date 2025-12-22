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

# --- 실행부 (데이터 원문 100% 반영: 한율) ---
if __name__ == "__main__":
    
    # ==========================================
    # 1. 설정: 브랜드 이름 변경
    # ==========================================
    current_brand_name = "한율"

    # 2. 데이터 입력 (보내주신 텍스트 전체 포함)
    
    # Part 1: 브랜드 철학 (K-Glass Skin & Balance)
    full_text_part1 = """
    나만의 건강한 균형
    Everyday glass skin, the K-beauty Recipe
    한율은 피부가 건강하게 빛나는 한국 여성의 스킨케어 비법을 전파하는 브랜드 입니다.
    매일 꾸준히 피부와 마음의 ‘비움’과 ‘채움’ 균형을 찾아 관리한 것이 본연의 피부 그 자체로 좋아보이는 한국 여성의 ‘K-glass Skin’의 비결이라고 한율은 생각합니다.
    혹독한 사계절의 변화를 견디며 생명력을 유지하는 한국 자연 원료의 응축된 힘을 피부에 순하지만 강력하게 전하는 동시에, 한국 발효 식품에서 찾아낸 성분을 마이크로바이옴 기술로 개발한 포스트바이오틱스 ‘Skin Balancer™’를 모든 제품에 담아 피부 즉각 개선은 물론 피부 건강을 위한 근본적 솔루션을 제공합니다.
    피부와 마음을 온전히 비우고 충분 채워 나만의 건강한 균형을 찾아주는 한율의 비움채움 루틴은 본질적으로 건강한 피부와 나의 본연의 빛이 드러나게 합니다.
    """

    # Part 2: 핵심 원료 및 헤리티지 (Local Ingredients & Heritage)
    full_text_part2 = """
    한국의 이로움으로 전하는 균형 잡힌 아름다움, 한율
    근본을 관리하는 건강한 피부와 생활 태도로
    균형 잡힌 아름다움을 전하는 K-heritage 클린 기능성 브랜드
    한국 로컬 원료의 이로움
    한국의 로컬 성분이 가진 힘을 극대화 하고 더 효과적으로 쓰일 수
    있도록 발전시키는 것,이것이 바로 한율의 시작입니다.
    쑥, 쌀, 콩. 일상에서 흔히 볼 수 있는 원료, 한율에게는 귀한 보물입니다.
    과거부터 전해온 선조들의 노하우가 현대의 우리에게도 뛰어난
    영감이자 솔루션이 될 수 있다고 믿기 때문입니다.
    한국 로컬에서 피부를 위한 잠재력을 가진 식물을 찾아 힘을 극대화 하고,
    더 효과적으로 쓰일 수 있도록 지속적인 연구를 통해
    피부 건강에 유효한 성분을 찾습니다.
    """

    # Part 3: 과학 기술 및 비전 (Microbiome & Active Balanced Beauty)
    full_text_part3 = """
    피부 스스로 강해질 수 있는 힘,
    마이크로바이옴 과학
    한국의 발효 식문화에는 자연스러운 미생물 반응으로 유익한 유기물을 발생시켜
    풍부한 영양을 담아내는 과학적 지혜가 담겨져 있습니다.
    한국의 발효 식문화에서 마이크로바이옴 과학의 실마리를 찾아 개발한 독자적 성분
    Skin Balancer™*.피부 장벽을 강화하여 외부 자극에 맞서 스스로
    강해질 수 있는 힘을 키워주는 한율 피부 과학의 결정체 입니다.
    *Skin Balancer™: 한국의 발효 식품에서 영감을 얻어 식물의 표면에서 발견한 락토바실러스 유산균을 배양한 포스트바이오틱스
    균형 잡힌 아름다움,
    Active Balanced Beauty
    건강한 피부 균형과 내면의 단단함을 통해 어떤 문제에도
    쉽게 무너지지 않는 균형 잡힌 아름다움에 진정으로 가까워 집니다.
    로컬 원료에서 찾은 고효능 성분으로 피부 표면의 문제를 즉시 해결함과
    동시에 근본의 힘을 길러 내외면의 균형을 이루게 합니다.
    나아가 피부에 편안한 제형, 기분 좋은 자연의 향,
    한국적 미학을 담은 패키지의 시각적 즐거움을 담아 모든 경험의
    순간에 안식과 힐링을 전하기 위해 노력합니다.
    지금은 한율과 함께
    아름다워지는 시간입니다
    아름다움의
    포인트를 발견하다
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