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

# --- 실행부 (데이터 원문 100% 반영: 에뛰드) ---
if __name__ == "__main__":
    
    # ==========================================
    # 1. 설정: 브랜드 이름 변경
    # ==========================================
    current_brand_name = "에뛰드"

    # 2. 데이터 입력 (보내주신 텍스트 전체 포함)
    
    # Part 1: 브랜드 비전 (Makeup Playlist & Culture)
    full_text_part1 = """
    Make Your Makeup Playlist! ETUDE!
    에뛰드는 즐거운 화장놀이 문화를 전파하는 대한민국 대표 메이크업 브랜드입니다.
    설레는 첫 메이크업의 순간을 함께하는 에뛰드는 메이크업이 더욱 쉽고 즐거운 경험이 되도록 최고의 제품과 서비스를 제공합니다. 다채로운 컬러, 사랑스러운 디자인을 통해 나만의 개성을 찾고, 아름다움을 발견하도록 도와줍니다.
    에뛰드는 화장놀이 문화를 전파하기 위해 에뛰드만의 메이크업 하우투를 생산하고, 경험을 확장합니다. SNS를 활용하여 다양한 미용법을 제공함은 물론, 뷰티 인플루언서가 되고 싶어하는 고객들을 대상으로 유튜버의 꿈을 이루어주는 '뷰티즌' 프로그램을 진행하고 있습니다. 이를 통해 뷰티를 통한 긍정적인 경험과 영향력을 발휘하고, 전 세계를 아우르는 즐거운 메이크업 문화를 만들어나갈 것입니다.
    """

    # Part 2: 브랜드 기원 및 가치 (Origin & Values)
    full_text_part2 = """
    color 라이프 스타일의 다채로움
    Pink 유니크한 핑크
    Princess 멋있고 자신감 있는 주인공
    Makeup Playlist
    나만의 스타일을 찾는 즐거운 화장놀이 문화
    ETUDE Comes from
    ‘ETUDE ’는 쇼팽이 작곡한 아름다운 연습곡의 이름에서 영감을 받아 탄생했어요.
    연습곡이 예술적인 멜로디로 변화하듯
    자신만의 아름다움을 찾아 변화할 수 있도록
    ETUDE는 신나고 설레는 메이크업의 모든 순간을
    함께 하고 싶다는 뜻을 담고 있어요.
    우리는 ETUDE 안에서 다양하고 생동감 넘치는 경험으로
    더 특별한 나를 표현할 수 있다고 믿어요!
    Since 1985
    1985년 12월 창립에서 시작된
    오랫동안 많은 분들의 사랑과 인정을 받아온 에뛰드 스테디셀러
    """

    # Part 3: 스테디셀러 제품 (Loved Products)
    full_text_part3 = """
    Loved Products
    자타공인, 역대 가장 많은 사랑을 받은,
    고민과 검증 끝에 출시된 제품을 만나보세요!
    Fixing Tint
    픽싱 틴트
    네이버 뷰티 윈도우 메이크업 10주 연속 1위
    *2021.11.09 기준
    2021년 글로우픽 어워드 위너 틴트 부문
    #안묻픽싱 #안묻틴트
    Curl Fix Mascara
    컬 픽스 마스카라
    글로우픽 10주 연속 1위컬링 마스카라 부문
    *2021.09.01 기준
    컬링 마스카라 5년 연속 1위 *칸타패널제공
    한국 색조 화장품 시장 내 컬링 마스카라 유형 2015.06.15~2020.06.14 구매량 기준 브랜드 랭킹
    국민BM 400명 검증 완료
    Moistful Collagen
    수분가득 콜라겐
    찐후기가 말해주는 믿고 쓰는 수분 크림
    수분감 만족도 98% *화해 200인 평가단
    보습력 만족도 97% *화해 200인 평가단
    Dear Darling Tint
    디어 달링 틴트
    오랜 시간 사랑받아온 #장수템 #애정템
    촉촉하고 선명하게 물드는 #워터젤 틴트
    2017년 아이스크림 틴트 #대란템
    Fixing Tint
    픽싱 틴트
    네이버 뷰티 윈도우 메이크업 10주 연속 1위
    *2021.11.09 기준
    2021년 글로우픽 어워드 위너 틴트 부문
    #안묻픽싱 #안묻틴트
    Curl Fix Mascara
    컬 픽스 마스카라
    글로우픽 10주 연속 1위컬링 마스카라 부문
    *2021.09.01 기준
    컬링 마스카라 5년 연속 1위 *칸타패널제공
    한국 색조 화장품 시장 내 컬링 마스카라 유형 2015.06.15~2020.06.14 구매량 기준 브랜드 랭킹
    국민BM 400명 검증 완료
    Moistful Collagen
    수분가득 콜라겐
    찐후기가 말해주는 믿고 쓰는 수분 크림
    수분감 만족도 98% *화해 200인 평가단
    보습력 만족도 97% *화해 200인 평가단
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