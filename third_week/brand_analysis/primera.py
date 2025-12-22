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

# --- 실행부 (데이터 원문 100% 반영: 프리메라) ---
if __name__ == "__main__":
    
    # ==========================================
    # 1. 설정: 브랜드 이름 변경
    # ==========================================
    current_brand_name = "프리메라"

    # 2. 데이터 입력 (보내주신 텍스트 전체 포함)
    
    # Part 1: SEED TECH 및 브랜드 정의
    full_text_part1 = """
    All about skin barrier by SEED TECH™
    씨드 테크 기반 피부 장벽 솔루션 브랜드 프리메라는 눈에 보이는 피부 문제를 개선하는 기존 스킨케어 방식에서 벗어나 오늘의 피부를 가장 건강하고, 가장 아름답게, 그리고 오래도록 유지하기 위해 Since 1986 씨드 연구 자산에 기반하여 피부 장벽의 근본적인 솔루션 ‘SEED TECHTM’를 제안합니다.
    프리메라 씨드 테크는 순수 고함량 식물 단백질에서 추출한 씨드 펩타이드의 피부 장벽 자생-리페어 효과에 피토바이오틱스의 장벽 방어-지속 효과가 더해져 건강하고 아름다운 피부를 완성합니다.
    """

    # Part 2: 믹솔로지 레시피 및 피부 고민 배경
    full_text_part2 = """
    Powerful Result Mixology Recipe
    Powerful Result Mixology Recipe
    Ⅰ “피부 고민이 한 가지만 있는 사람은 없다”
    우리는 칙칙한 피부 톤, 늘어진 모공, 요철, 과다한 피지 등
    복합적인 피부고민을 가지고 있습니다.
    Ⅱ “90% 이상이 느끼는 민감 피부”*
    대한민국 여성 90% 이상이 본인 피부가 “민감”하다고 생각합니다.
    그만큼 대다수 여성이 피부 자극에 두려움을 가지고 있습니다.
    * 자사 민감고객리포트: 20-35세 여성, 500명 온라인 설문조사
    프리메라는 성분/제형/유형 mix을 통해
    복합적인 피부고민 솔루션을 저자극 제형에 담은 고효능 X 저자극의 믹솔로지 브랜드 입니다.
    """

    # Part 3: 구체적 솔루션 및 효능 (비타민C, 레티놀 등)
    full_text_part3 = """
    Powerful Result Mixology Recipe
    투명도, 탄력, 민감, 건조, 과다 피지 등의
    타겟화된 피부고민을
    고효능 x 저자극 솔루션으로 눈에 띄게 개선
    임상으로 증빙 된 파워풀한 피부 개선 효능
    스킨케어 전 제품 피부과 테스트/ 1차 자극 테스트 완료
    프리메라는 스킨케어 효능 성분 / 제형 / 유형의
    여러 가지 조합을 통해, 효율적으로
    복합적인 고민을 해결하는
    파워풀한 믹솔로지 솔루션을 제안합니다
    비타민C X 레티놀 등 효능 성분의
    새로운 조합으로 복합 피부 고민 개선
    오일투폼, 필링투폼 등 다양한 제형의 조합으로 루틴 혁신
    레티놀 립세럼 등 스킨케어와 메이크업 결합한 하이브리드 메이크업
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