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

# --- 실행부 (데이터 원문 100% 반영: 헤라) ---
if __name__ == "__main__":
    
    # ==========================================
    # 1. 설정: 브랜드 이름 변경
    # ==========================================
    current_brand_name = "헤라"

    # 2. 데이터 입력 (보내주신 텍스트 전체 포함)
    
    # Part 1: 브랜드 철학 (Contemporary Seoul Beauty)
    full_text_part1 = """
    Contemporary Seoul Beauty
    Here, Now, Myself
    지금, 여기에서, 가장 나답게.
    헤라는 서울의 역동적인 아름다움에 주목하고 이를 더욱 다채로운 나다움으로 재해석하여 세계에 전파하는 컨템포러리 서울 뷰티 브랜드입니다.
    헤라가 추구하는 아름다움에 대한 신념은 나 자신의 본질에 있습니다. 모두가 고유한 자신만의 아름다움을 발현할 수 있도록, 늘 한 발 앞선 새로운 해석과 시도를 통해 서울리스타의 아름다움을 시대 정신에 맞게 발전시키고 있습니다.
    서울리스타의 아름다움, 그 중심엔 자기주도적이고 능동적인 삶의 태도가 있습니다. 빠르게 변화하는 도시 속에서 자신의 모든 가능성을 열어두고 한계를 제한하지 않으며, 깨어있는 감각과 유연한 사고 방식으로 자신만의 아름다움을 탄생시켜 나갑니다.
    본연의 아름다움을 발산하는 베이스부터 다양성을 포용하는 편안한 컬러 메이크업까지, 헤라의 뷰티 루틴은 고유의 아름다움을 있는 그대로 드러내며 발산되는 존재감으로 나만의 다채로운 아름다움을 발굴하게 합니다.
    """

    # Part 2: 대표 제품 라인업 (Product Lineup)
    full_text_part2 = """
    SENSUAL POWDER MATTE LIQUID
    BLACK CUSHION FOUNDATION
    SIGNIA ILLUMINATING SERUM
    COMFY REVITALIZING SERUM MIST
    ZEAL EAU DE PARFUM FOR WOMEN
    """

    # Part 3: 서울리스타 및 뷰티 루틴 (Seoulista & Routine)
    full_text_part3 = """
    CONTEMPORARY
    SEOUL BEAUTY
    헤라는 서울의 역동적인 아름다움에 주목하고 이를 더욱 다채로운 나다움으로
    재해석하여 세계에 전파하는 컨템포러리 서울 뷰티 브랜드입니다.
    헤라가 추구하는 아름다움에 대한 신념은 나 자신의 본질에 있습니다.
    모두가 고유한 자신만의 아름다움을 발현할 수 있도록,
    늘 한 발 앞선 새로운 해석과 시도를 통해 서울리스타의 아름다움을
    시대 정신에 맞게 발전시키고 있습니다.
    서울리스타의 아름다움, 그 중심엔 자기주도적이고 능동적인 삶의 태도가 있습니다.
    빠르게 변화하는 도시 속에서 자신의 모든 가능성을 열어두고 한계를 제한하지 않으며,
    깨어있는 감각과 유연한 사고 방식으로 자신만의 아름다움을 탄생시켜 나갑니다.
    유연하지만 견고하게,
    나만의 다채로운 아름다움을 발굴하는 헤라의 뷰티 루틴
    본연의 아름다움을 발산하는 베이스부터 다양성을 포용하는 편안한 컬러 메이크업까지
    고유의 아름다움을 있는 그대로 드러내며 발산되는 존재감을 의미합니다.
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