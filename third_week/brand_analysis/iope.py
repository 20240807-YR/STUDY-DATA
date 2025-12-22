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

# --- 실행부 (데이터 원문 100% 반영: 아이오페) ---
if __name__ == "__main__":
    
    # ==========================================
    # 1. 설정: 브랜드 이름 변경
    # ==========================================
    current_brand_name = "아이오페"

    # 2. 데이터 입력 (보내주신 텍스트 전체 포함)
    
    # Part 1: 브랜드 정의 (Clinical Level Tech & Bio Science)
    full_text_part1 = """
    클리니컬 레벨 테크(Clinical Level Tech)로 눈에 보이는 변화를 주는 고효능 스킨케어 브랜드
    국내 기능성 화장품 시장을 대중화한 아이오페는 바이오 사이언스(Bio Science) 근간의 효능성분(Bio Origin)과 전달기술(Bio Tech) 연구를 통해 확실하게 검증한 고효능 솔루션을 제공합니다.
    아이오페는 단순한 피부표면의 일시적 개선이 아닌, 피부 겉부터 속까지 피부 전체에 작용하는 효능의 근원적 솔루션을 선사합니다.
    이러한 아이오페만의 클리니컬 레벨 테크(Clinical Level Tech) 고효능 제품을 통해 눈에 보이게 좋아지는 피부 변화를 체감할 수 있으며, 본연의 아름다움과 자신감을 찾아 보다 건강하고 활력 넘치는 삶의 변화를 이끌어줍니다.
    """

    # Part 2: 연구소 및 정신 (IOPE LAB & LAB SPIRIT)
    full_text_part2 = """
    IOPE LAB
    아이오페 랩은 브랜드의 오리진으로 피부와 소재 그리고 기술에 대한 연구가 이루어지는 공간입니다.
    브랜드가 탄생한 1996년부터 현재까지, 전문 연구원인 ‘아이오페 Innovator’들을 통해 도전적 연구가 끊임 없이
    진행중이며 이는 아이오페 만의 혁신적인 성과와 특별한 자산으로 축적되고 있습니다.
    LAB SPIRIT
    피부의 미래를 바꾸는 연구 정신
    더 나은 피부 미래를 위한 도전적인 연구 정신은 아이오페만의 혁신적인
    독자 기술과 특허로 그 성과를 증명하고 있습니다.
    깊이 있는 피부 연구
    실제 피부 분석과 차별화된 기술 연구로
    고효능 성분을 효과적이고 안전하게 피부에
    전달합니다.
    모든 과정을 심도 있게
    아이오페는 피부부터 소재, 기술까지
    모든 과정에 걸쳐 심도있는 연구를 합니다.
    고객 맞춤 솔루션
    개인마다 다른 피부 고민과
    타고난 피부까지 고려한 맞춤 솔루션은
    피부의 미래를 바꾸어 줍니다
    """

    # Part 3: 가치 및 관계 (Essence & Benefit)
    full_text_part3 = """
    ESSENCE & BENEFIT
    피부미래연구소
    아이오페랩은 아이오페 이노베이터와 고객간의 직접적인 만남이 이루어지는 공간입니다.
    고객은 이 곳에서 피부 솔루션을 스스로 탐색할 수 있고, 나아가 전문가 진단을 통해 최적의 솔루션을 제안 받을 수도 있습니다.
    또한 이노베이터는 고객과의 직접적인 만남을 통해 새로운 연구에 대한 영감을 얻고 더 나은 솔루션을 위한 기회를 탐색합니다.
    피부 전문가
    IOPE INNOVATOR
    고객 맞춤 솔루션
    고객
    CUSTOMER
    다양한 연구 기회 제공
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