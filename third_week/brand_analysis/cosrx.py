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

# --- 실행부 (데이터 원문 100% 반영: 코스알엑스) ---
if __name__ == "__main__":
    
    # ==========================================
    # 1. 설정: 브랜드 이름 변경
    # ==========================================
    current_brand_name = "코스알엑스"

    # 2. 데이터 입력 (보내주신 텍스트 전체 포함)
    
    # Part 1: 브랜드 철학 (Cosmetics + Rx)
    full_text_part1 = """
    오직 성분으로, 오직 코스알엑스
    "Cosmetics(화장품) + Rx(처방) = COSRX"
    코스알엑스는 피부 타입과 관련된 문제를 넘어 우리의 일상이 더욱 아름다워질 수 있도록 하는 것이 무엇인지를 고민해왔습니다.
    화장품이 가져야 할 본질을 들여다보고 아름다움 본연의 가치를 담은 제품을 통해 소비자분들을 만나고 있습니다.
    연구 기반 최적의 성분 배합으로 전 세계 소비자의 피부 문제와 고민은 덜어내고, 진심과 안심을 더하기 위해 노력합니다.
    """

    # Part 2: 브랜드 페르소나 (Mr. Rx)
    full_text_part2 = """
    미스터 알엑스는 오랜시간 동안 일상 속 뷰티 라이프 스타일을 면밀히 바라보고, 피부타입과 관련된 문제를 넘어 우리의 일상이
    더욱 아름다워질 수 있도록 하는 것이 무엇인 지를 고민해왔습니다. 또한 뷰티 산업에서의 다년간의 노하우와 경험을 바탕으로
    코스알엑스라는 이름으로 소비자들을 만나오고 있습니다. 미스터 알엑스는 우리 모두를 대변하며 코스알엑스의 상징으로서
    많은 사람들이 본연의 아름다움을 찾길 바라는 마음으로 정성을 다하고 있습니다.
    """

    # Part 3: 브랜드 컨셉 및 키워드 (Soft Skin Care & Solution)
    full_text_part3 = """
    brand concept Soft Skin Care
    ayered for (sensitive) skin/] Fashion / Fabric - Feeling
    [/texture & tactile sense/
    T.P.O / R(X) = Solution
    (layered a edit a tailor-made/]
    민감피부를위한저자극스킨케어
    피부친화적/저자극/부드러운텍스처
    라이프스타일/일상/공감
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