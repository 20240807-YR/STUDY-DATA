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

# --- 실행부 (데이터 원문 100% 반영: 에스트라) ---
if __name__ == "__main__":
    
    # ==========================================
    # 1. 설정: 브랜드 이름 변경
    # ==========================================
    current_brand_name = "에스트라"

    # 2. 데이터 입력 (보내주신 텍스트 전체 포함)
    
    # Part 1: 브랜드 헤리티지 및 의미 (Heritage & Estuary)
    full_text_part1 = """
    태평양제약에서 시작된 헤리티지에 기반해 오래도록 건강한 아름다움을 선사하는 브랜드
    태평양제약으로 시작된 헤리티지에 기반하여, 오래도록 건강한 아름다움을 선사하는 브랜드, 에스트라
    강과 바다가 만나 이뤄낸 가장 비옥한 땅, 삼각주처럼
    에스트라는 태평양 제약으로부터의 헤리티지와 아모레퍼시픽의 최첨단 피부과학의 만남을 근간으로,
    피부과 전문의들과의 협업을 지속하며 시대에 따라 달라지는 민감피부의 고민해결을 위해
    겉으로 드러난 문제 뿐 아니라 그 근본 원인까지 총체적으로 케어하는 궁극의 피부 솔루션을 제공하며
    건강한 피부를 통해 찾아가는 충만하고 자신감 있는 삶, 오래도록 건강한 아름다움을 선사합니다.
    AESTURA”라는 브랜드명은 영어 단어 “ESTUARY”에서 유래되었으며,
    이는 라틴어 단어 “aestuarium”에서 온 조수의 영향을 받는 해안을 의미합니다.
    강과 바다가 만나 이루어 낸 비옥한 삼각주처럼,
    브랜드는 1982년에 설립된 전신 PACIFIC PHARMA Co.에서 시작된 경험과 연구를 통해
    건강한 아름다움을 구현하고 피부 고민 개선에 기여하는 것을 목표로 합니다.
    """

    # Part 2: 의학적 전문성 및 MD 인증 (Doctors, No.1 & Medical Device)
    full_text_part2 = """
    에스트라가 의사를 만나는 이유
    일시적 윤기와 잠깐의 아름다움이 아닌
    근본적인 피부의 건강과 개선을,
    의사와 함께 고민해나갑니다.
    민감 피부는 학술적으로 정의할 수 있는 질환이 아니라,
    피부에 다양한 불편함을 느끼는 주관적인 느낌이자
    멘탈적인 상태이기 때문에 그래서 에스트라는
    스스로 민감 피부 전문가가 되기로 했습니다.
    오랜 연구 끝에, 민감 피부를 ‘피부 장벽이 약해진 상태’로 정의하고,
    민감 피부의 건조나 노화는 일반 정상 피부와 어떻게 다른지 각종 논문 지식을 섭렵했습니다.
    같은 민감피부라도 저마다 느끼는 문제의 지점이 다릅니다.
    이에 에스트라는 51명의 피부과 전문의와 최신 피부과학의 지견을 나누며
    인사이트를 얻고 끊임없이 연구합니다.
    9년 연속 No.1 한국 병원 화장품 브랜드
    한국소비자브랜드 위원회, 2016-2024
    병원 화장품 부문 올해의 브랜드 대상
    한국 상급종합병원 에스트라 더마 솔루션 채택
    국내 49개 상급종합병원 처방,
    4100여개의 종합병원 의원 입점
    건강심사평가원(HIRA) 수치 기준. 의료기기(MD) 한정.
    메디컬 디바이스(MD) 인증을 받은 이유
    에스트라는 MD 제조 인증을 받는 것이 까다롭고 매우 어려운 과정이라는 것을
    알면서도 늦출 수 없었습니다.
    에스트라는 아토피피부염 환우를 포함하여 보습제 사용이 필수적인
    문제성 피부 환자들의 고민이 얼굴뿐만 아니라 전신으로 발현된다는
    사실을 알게 되었습니다. 피부를 회복하기 위해서는 보습제를 충분히
    사용해야 하는데, 넓은 면적에 사용해야 하는 환자들에게는 상당한
    비용 부담이 따랐습니다.
    아토피 피부염 등 피부 장벽이 손상된 피부의 솔루션을 위해
    MD는 의료기기를 뜻하는 메디컬 디바이스(Medical Device)의 약자입니다.
    MD 제품은 일반 화장품 회사에서는 생산할 수 없습니다.
    식약처로 부터 의료기기 제조업 허가를 획득해야 하고,
    의료기기 제조에 준하는 엄격한 제조 공정과 품질 검수 등
    일정한 기준과 절차에 부합해야 합니다.
    신청 후 인증을 받는 데까지 수년이 걸리는 일이기도 합니다.
    에스트라는 MD 제조 인증을 받는 일이 매우 까다롭고
    어려운 과정임을 알면서도 늦출 수 없었습니다.
    2018년 아토베리어 라인에 이어
    2022년 더마베이비 프로 라인에 MD 인증을 받았습니다.
    """

    # Part 3: 유통 확장 및 365 라인 (Retail Expansion & Product Lines)
    full_text_part3 = """
    병원 밖으로 나온 더마 with 올리브영
    에스트라는 병원 채널을 유지하며
    ‘시판’이라는 새로운 도전에 나서게 됩니다.
    2016년 에스트라는 유수의 해외 더모코스메틱 브랜드들을 제치고
    ‘ 올해의 브랜드’ 병원화장품 부문 대상을 차지했습니다.
    2017년에는 특별한 바이럴 활동을 하지 않았음에도
    아토베리어 크림이 순수하게 병원 방문 고객의 후기만으로
    ‘화해 뷰티 어워드’ 크림·젤 부문 1위에 올랐습니다.
    병원화장품은 병원에서만 살 수 있다 보니 병원에 간 김에 화장품을 살 수는 있지만,
    화장품을 사러 일부러 병원에 가기는 쉽지 않다는 이야기였습니다.
    이미 에스트라는 아토베리어 MD 라인 출시 과정에서 더 많은 고객의
    피부고민을 해결하고자 문제성 피부에서 민감성 피부로
    솔루션 폭을 확대하기로 했던 터였습니다.
    고객과의 접점을 확장해야 한다는 결론에 이르게 했습니다.
    ‘이제 병원 밖 일상에서도 고객을 만나자’
    에스트라는 병원 판매 채널을 유지하며
    ‘시판’이라는 새로운 도전에 나섰습니다.
    시판에 앞서 에스트라는 문제성 피부고민에 맞추어
    병원에서 유통하던 아토베리어(보습),
    테라크네(트러블), 리제덤(자생력 강화) 등
    세 라인을 민감성 피부에 맞게 다시 연구하고 개발해
    시판용 라인을 완성했습니다.
    시판용 라인에는 ‘데일리 더마솔루션’이라는 의미를 더해
    라인명 뒤에 365를 붙였습니다.
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