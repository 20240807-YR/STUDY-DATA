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

# --- 실행부 (데이터 원문 100% 반영: 라네즈) ---
if __name__ == "__main__":
    
    # ==========================================
    # 1. 설정: 브랜드 이름 변경
    # ==========================================
    current_brand_name = "라네즈"

    # 2. 데이터 입력 (보내주신 텍스트 전체 포함)
    
    # Part 1: 브랜드 철학 및 호기심 (Curiosity & Philosophy)
    full_text_part1 = """
    아름다움을 향한 끝없는 호기심
    OPEN TO WONDER.
    호기심.
    무한한 가능성을 담은 세 개 의 글자.
    호기심. 이 짧은 단어 하나가
    이전에 없던 스킨케어를 만듭니다.
    우리를 놀라게 하고,
    우리의 상상력을 자극하며,
    우리의 마음을 엽니다.
    그리고 마침내 누구도 한 적 없는 도전을 하게 만듭니다.
    만약에, 크림 한 통을 통째로 녹여 토너를 만들면 어떨까?
    만약에, 립밤과 슬리핑 마스크를 섞어 하나로 합친다면?
    만약에, 히알루론산을 잘게 쪼개 피부 깊숙이 스며들게 한다면?
    호기심에는 힘이 있습니다.
    복잡한 것을 쉽고 편하게 만드는 힘.
    세상의 규칙으로부터 우리를 더욱 자유롭게 만드는 힘.
    우리를 우리답게 만드는 힘.
    하지만, 호기심은 시작일 뿐,
    진정한 놀라움은 아직 시작되지 않았습니다.
    우리가 그 문을 직접 열기 전까지는.
    아름다움을 향한 끝없는 호기심
    Skincare driven
    by curiosity and science
    OPEN TO WONDER.
    라네즈는 “만약에”라는 질문을 통해
    끝없는 호기심을 품고, 더 나은 스킨케어의 가능성에 대해 확신을 가지며,
    아름다움의 경계를 계속해서 넓혀가고 있습니다.
    라네즈와 함께 놀라운 경험의 세계로 나아가세요.
    아름다움을 향한 끝없는 호기심
    OPEN TO WONDER.
    호기심.무한한 가능성을 담은 세 개 의 글자.
    호기심. 이 짧은 단어 하나가 이전에 없던 스킨케어를 만듭니다.우리를 놀라게 하고, 우리의 상상력을 자극하며, 우리의 마음을 엽니다.
    그리고 마침내 누구도 한 적 없는 도전을 하게 만듭니다.
    만약에, 크림 한 통을 통째로 녹여 토너를 만들면 어떨까?
    만약에, 립밤과 슬리핑 마스크를 섞어 하나로 합친다면?
    만약에, 히알루론산을 잘게 쪼개 피부 깊숙이 스며들게 한다면?
    호기심에는 힘이 있습니다. 복잡한 것을 쉽고 편하게 만드는 힘. 세상의 규칙으로부터 우리를 더욱 자유롭게 만드는 힘. 우리를 우리답게 만드는 힘.
    하지만, 호기심은 시작일 뿐, 진정한 놀라움은 아직 시작되지 않았습니다. 우리가 그 문을 직접 열기 전까지는.
    """

    # Part 2: 핵심 제품군 및 솔루션 (Hydration & Efficacy)
    full_text_part2 = """
    THE CORE PRODUCT PILLARS
    HARNESSING HYDRATING WONDERS
    모든 피부 문제는 수분 부족으로부터 시작 됩니다. 피부의 가장 기본적이면서 핵심적인 문제인 ‘수분 부족’을 해결하기 위해 라네즈는 단순한 보습을 넘어선 스킨케어의 혁신을 목표로 삼았습니다. 라네즈는 피부 표면에 수분을 공급할 뿐만 아니라 피부의 속 보습을 유지하는 데 도움을 주는 기술과 혁신을 만들고 있습니다. 이런 수분 공급에 초점을 맞춘 두 가지 핵심 제품인 워터뱅크 라인와 크림 스킨 토너는 피부에 수분을 제공하여 보습력을 유지시켜주며 모든 피부 유형이 사용할 수 있는 제형의 혁신을 적용했습니다. 라네즈가 끊임없는 혁신으로 만들어낸 수준 제품을 통해 기존 화장품에서 경험할 수 없었던 수분 보습과 장벽 강화를 경험해보세요.
    EFFICACY-DRIVEN SOLUTIONS
    SCIENCE-BACKED SKINCARE WONDERS
    라네즈의 새로운 기술은 피부 노화에 따른 전반적인 피부 건강의 보호 및 강화에 중점을 두고 있습니다.
    이러한 기술을 기반으로 라네즈에서는 바운시 앤 펌, 래디언-C, 퍼펙트 리뉴라는 3가지 효과 중심 제품 라인을 탄생시켰습니다. 각 라인은 주름완화 (퍼펙트 리뉴), 탄력강화 (바운시 앤 펌), 광채 (래디언-C)를 위한 강력한 효과를 지닌 제품으로 고객들이 원하는 최고의 효능을 선보입니다.
    라네즈의 기술은 항상 눈에 보이는 뛰어난 개선 효과를 제공하는 것을 목표로 하며, 건강한 피부를 유지 및 회복하고자 하는 고객을 위한 최고의 솔루션이 될 것입니다.
    """

    # Part 3: 슬리핑 뷰티 및 메이크업 (Sleeping Beauty & Makeup)
    full_text_part3 = """
    SLEEPING BEAUTY TECHNOLOGY
    PIONEERING OVERNIGHT SKIN WONDERS
    슬리핑 뷰티 카테고리의 선구자인 라네즈는 수면시간을 활용한 나이트 스킨케어의 역할을 새롭게 정의했습니다. 슬리핑 마스크 시리즈는 자는 동안 피부에 풍부한 영양과 휴식을 제공하고 활력을 불어넣을 수 있도록 얼굴, 입술, 눈가를 관리할 수 있는 각각의 제품으로 설계되었습니다. 특히 스킨케어의 골든 타임인 수면 시간 동안 경험할 수 있는 수분(워터), 진정(시카), 탄력(펩타-콜라겐) 슬리핑 마스크의 솔루션을 제공합니다.
    제품을 경험 한 다음날 아침이면 완벽하게 개선 된 피부를 경험 할 수 있습니다.
    THE POWER OF SKINCARE, THE IMPACT OF MAKEUP
    라네즈의 하이브리드 라인의 혁신은 스킨케어 제품과 메이크업 제품의 장점을 결합하는 데 있습니다.
    하이브리드 스킨케어의 대표주자인 스킨 베일 베이스와 네오 쿠션은 스킨케어 효능으로 피부를 보호하면서 동시에 완벽한 커버력과 아름다운 피부 톤을 제공합니다.
    기존 화장품의 효과를 넘어서 아름다운 피부를 연출함과 동시에 실제로 피부 건강을 유지하는 데 도움을 주는 제품 라인입니다.
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