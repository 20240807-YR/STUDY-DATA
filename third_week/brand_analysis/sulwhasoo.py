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

# --- 실행부 (데이터 원문 100% 반영: 설화수) ---
if __name__ == "__main__":
    
    # ==========================================
    # 1. 설정: 브랜드 이름 변경
    # ==========================================
    current_brand_name = "설화수"

    # 2. 데이터 입력 (보내주신 텍스트 전체 포함)
    
    # Part 1: 브랜드 철학 (Holistic Beauty & Journey)
    full_text_part1 = """
    시간의 흐름에 지지 않는 아름다움
    Journey to Holistic Beauty
    설화수는 전통의 지혜에서 영감을 받고 최신의 현대기술을 접목하여 솔루션을 만듭니다.
    한국의 미학에 동시대적 시선을 더해 우리만의 아름다움을 완성해갑니다.
    겉으로 드러나는 문제 해결 뿐 아니라, 눈에 보이지 않는 본질까지 더 넓고 깊게 연구합니다.
    그러므로 설화수의 홀리스틱뷰티란 부분이 아닌 전체를 보는 시선, 숨겨진 본질에 대한
    사유, 과거와 미래의 시공간을 초월하는 진화입니다.
    끝없이 경계를 허물고 가능성을 확장하는 우리의 여정은
    시간의 흐름에 지지 않고 계속됩니다.
    자음생 브랜드 스토리 필름
    홀리스틱 뷰티
    피부 장수를 위한 인삼
    피부 능력을 키우는 자음단®
    윤빛 도는 힘있는 피부
    OUR JOURNEY
    시간의 흐름에 지지 않는 아름다움
    홀리스틱 뷰티
    설화수는 전통의 지혜에서 영감을 받고 최신의 현대기술을 접목하여 솔루션을 만듭니다.
    한국의 미학에 동시대적 시선을 더해 우리만의 아름다움을 완성해갑니다.
    겉으로 드러나는 문제 해결 뿐 아니라, 눈에 보이지 않는 본질까지 더 넓고 깊게 연구합니다.
    그러므로 설화수의 홀리스틱뷰티란 부분이 아닌 전체를 보는 시선, 숨겨진 본질에 대한
    사유, 과거와 미래의 시공간을 초월하는 진화입니다.
    끝없이 경계를 허물고 가능성을 확장하는 우리의 여정은
    시간의 흐름에 지지 않고 계속됩니다.
    """

    # Part 2: 핵심 원료 및 연구 (Ginseng & JAUM Activator)
    full_text_part2 = """
    집념의 60년 연구,
    피부 장수를 위한 인삼
    먹어서 좋다면 피부에 발라도 좋지 않을까?
    우리 땅에서 나는 원료를 넣은 글로벌 제품을 만들고자 우수한 효능을 인정받아온
    귀한 원료, 고려인삼을 스킨케어에 접목한 우리의 연구는 1964년에 시작되었습니다.
    에이징케어를 넘어 피부 장수(Skin Longevity)까지 고려인삼이 지닌 생명력을
    피부에 오롯이 전하기 위한 60여 년 간의 인삼 연구는
    첨단 과학 기술을 접목한 독자 성분 개발로 이어져 오고 있습니다.
    인삼의 뿌리부터 줄기, 잎, 열매까지
    설화수의 독자적인 인삼 연구는 계속됩니다.
    5종 식물 원료 조합의 시너지
    피부 능력을 키우는 자음단®
    홀리스틱 뷰티를 추구하는 설화수의 철학을 담은 독자 성분, 바로 자음단®입니다.
    자음단®은 여러 성분들이 서로 보호, 보완하며 시너지를 낼 수 있도록 조화롭게 구성하는
    복합적인 배합을 기본으로 합니다.
    안전하고 강력한 효능을 낼 수 있는 가공법과 조합을 수도 없이 테스트하여 작약, 연자육,
    옥죽, 백합, 지황 5종의 식물 원료를 엄선, 설화수만의 고유한 처방이자 기본이 되는
    자음단®이 탄생했습니다.
    자음단®은 피부 본연의 건강한 능력을 활성화하여 장벽 강화, 보습, 탄력을 개선시켜줍니다.
    자음단® 상세 영상
    """

    # Part 3: 피부 효능 및 제품 라인업 (Skin Benefits & Products)
    full_text_part3 = """
    설화수가 만드는
    윤빛 도는 힘 있는 피부
    우리 피부는 환경과 라이프스타일에 의해 매일 미세한 노화의 자극들을 겪고 있습니다.
    이런 자극들과 노화 증상 개선을 위해 피부는 스스로 이겨낼 수 있는 힘,
    즉 자생력을 가져야 합니다.
    설화수는 모든 제품 효능의 기본을 자생력으로 삼고 각 제품별 구체화된 피부 개선
    효과들을 더했습니다.
    정체된 피부 흐름을 되살려 윤빛 도는 피부를 완성하는 윤조에센스.
    피부 기둥을 세워 자생력으로 차오른 고밀도 피부를 선사하는 자음생크림.
    노화 흔적을 지우고 숨겨진 피부 각도를 되찾아주는 진설크림.
    끝없이 경계를 허물고 가능성을 확장하는 우리의 여정은
    시간의 흐름에 지지 않고 계속됩니다.
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