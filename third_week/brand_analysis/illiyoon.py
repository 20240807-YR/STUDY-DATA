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

# --- 실행부 (데이터 원문 100% 반영: 일리윤) ---
if __name__ == "__main__":
    
    # ==========================================
    # 1. 설정: 브랜드 이름 변경
    # ==========================================
    current_brand_name = "일리윤"

    # 2. 데이터 입력 (보내주신 텍스트 전체 포함)
    
    # Part 1: 브랜드 철학, 기준 및 연구소 (Philosophy & Criteria)
    full_text_part1 = """
    Amorepacific
    민감한 피부 고민에 대한 순하고 강력한 답, 일리윤
    일리윤(ILLIYOON, 一理潤)은 40여년의 민감 피부 연구를 통해 얻은 피부와 닮은 보습 특화 성분으로 민감한 피부 고민을 근본적으로 해결하는 저자극 더마 보습 브랜드입니다.
    민감은 피부만의 문제로 볼 수 없기 때문에, 일리윤은 피부 효능 뿐 아니라 환경, 생애 주기까지 복합적으로 연구하여 피부 본연의 건강한 힘을 되찾을 수 있는 순하고 강력한 답을 제시합니다. 민감한 피부를 가진 고객이 불편하지 않도록 오롯이 민감 피부만 생각해 온 진심을 더합니다.
    민감한 피부가 삶을 방해하지 않도록,
    오롯이 민감 피부만 생각해 온
    진심이 있습니다.
    일리윤은 40여년의 다각적인 연구를 바탕으로
    피부고민을 근본적으로 해결할 수 있도록
    민감 피부에 순하고 강력한 답을 제시합니다.
    까다로운 한국인의 민감 피부 고민을
    해결해 온 브랜드는 따로 있어요.
    한국인은 피부 장벽이 얇아 자극에 쉽게 예민해지고,
    변화무쌍한 사계절과 미세먼지 같은 다양한 자극에 노출되어 있습니다.
    이처럼 민감 피부를 유발하는 요소와 상황들이 무척 다양하기 때문에
    화장품의 성분, 제조 공정, 사용감까지 꼼꼼하게 체크하여 제품을 선택합니다.
    일리윤은 오랜 기간 쌓아온 데이터를 바탕으로
    민감 피부를 위한 5가지 기준을 선정하여,
    한국인에 특화된 민감 피부 솔루션을 제시합니다.
    한국형 민감 피부를 위한
    다섯 가지 기준
    1. 피부 장벽 강화 : 유전적으로 얇은 한국인의 피부 장벽을 튼튼하게 강화
    2. 순하지만 강력한 효능: 화장품 성분에 대한 높은 지식과 관여도를 만족시켜줄 순하지만 강력한 효능
    3. 믿을 수 있는 공정: 원료 추출부터 제조까지 체계적이고 믿을 수 있는 공정
    4. 순한 저자극의 사용감 : 화장품 사용감도 민감의 원인이 될 수 있기에 자극을 줄이는 까다로운 테스트
    5. 피부 본연의 힘 : 거친 외부 환경도 스스로 이겨낼 수 있도록 피부 본연의 건강함을 회복
    일리윤의 민감연구소
    한국인 민감 피부 연구의 시작과 끝
    A. 피부 연구
    건강한 피부와 다른 민감 피부만의 차이점과 각질층 특성까지 파악하여 보이는 피부 고민은 물론, 보이지 ㅇ낳는 근본적인 원인과 해결책을 함께 찾습니다. 
    B. 라이프 연구
    민감함은 피부만의 문제가 아니기에 인종, 연령, 기후, 환경 별 특징 등 민감함의 원인이 되는 요소는 물론 생활 습관까지 다각도로연구합니다. 
    C. 민감성 연구
    일리윤은 체계적이고 까다로운 통과 기준을 통해 순수하고 효과적인 솔루션을 제공합니다. 
    D. 제형 연구
    피부에 닿는 자극을 줄일 수 있도록 촉촉함이 하루종일 지속되도록 피부 속 깊숙한 곳까지 제대로 흡수되는 제형을 만듭니다. 
    """

    # Part 2: 고객 및 연구원 인터뷰 (Interviews)
    full_text_part2 = """
    INTERVIEW 01
    모윤희
    (17, 고등학생)
    어릴 때부터 피부 가려움* 등 다양한 피부고민이 심해서
    피부가 무척 예민해요.
    핸드크림, 바디로션 하나를 사더라도 후기부터 성분까지
    습관적으로 꼼꼼하게 확인해야 했구요.
    청소년 **시기부터 미리 케어해야 한다고
    엄마가 추천해준 첫 더마제품이 ‘일리윤’이었어요.
    * 건조로인한 가려움
    ** 만 14세 이상의 청소년
    까탈스러운 제 피부에
    효과가 있었다는 것만으로도
    꾸준히 사용 할 이유는
    충분한 것 같아요.
    INTERVIEW 02
    오다연
    (20, 대학생)
    환절기마다 피부가 뒤집어지는 전형적인
    민감 피부를 가지고 있어요.
    마스크 때문에 피부가 자극 받아서 유수분 밸런스도 깨지고
    모공도 넓어진 상태 였거든요.
    확실히 트러블*이 진정되면서
    피부 컨디션이 달라졌구요,
    발림성 좋은 저자극에
    흡수력이 좋아 밀림이 없다 보니
    화장 잘 먹는 피부로 바뀌었어요. * 건조로인한 가려움
    INTERVIEW 03
    최은수 (32)
    강태욱 (35)
    강민채 (4)
    출산 후 몸이 예민해져서 온 가족이 쓸
    제품을 찾던 중 일리윤을 알게 되었어요.
    아이와 함께 쓸 제품이니 더 까다롭게 골랐죠.
    무향에 색소도 들어가지 않아서
    아이들은 물론 온 가족이
    편안하게 쓸 수 있는게 큰 장점이에요.
    특히 세라마이드 아토 집중 크림은 얼굴부터 몸까지 한 번에 사용하기도 좋아서 아이들 씻기고 쓰기 너무 편하고, 효과도 확실해요
    INTERVIEW 04
    김서영 (42)
    일리윤 연구원
    민감 피부는 환경과 계절 등 다양한 요소에 따라 지속적으로
    영향을 받기에 평생 관리의 대상이 될 수 밖에 없습니다.
    일리윤이 고강도의 테스트를 통해 민감함을 유발할 수 있는
    다양한 환경과 상황을 연구하는 이유도 바로 이 때문이죠.
    민감 피부는 더이상 일부의 성별과 특정 연령의
    고민이 아닙니다.
    온 가족 모두가 믿고 사용할 수 있으려면
    순한 제형은 기본이며, 강력한 효능이 보장되어야 합니다.
    일리윤은 지속적인 연구개발과 데이터를 바탕으로 확인된 제품을 완성해왔고, 꾸준히 사랑받아 왔습니다
    고객 한 분 한 분이 바로 일리윤의 결과이며, 우수한 제품을 꾸준히 만들어 낼 수 있는 원천이 되어 주셨습니다
    누구라도 한 번 써보면 일리윤의
    진정한 팬이 될 수 밖에 없을 겁니다.
    앞으로 더 많은 분들이
    일리윤의 가치와 우수한 제품력을
    누릴 수 있었으면 합니다.
    왜 민감 피부에는
    세라마이드가 필요할까요?
    일리윤은 오랜 연구를 통해 민감 피부의 공통점을 발견하게 되었습니다.
    바로 피부 지질의 절반을 차지하는 세라마이드 성분이 감소했다는 것입니다.
    세라마이드가 부족하면 피부 장벽이 쉽게 손상되어, 외부환경에 자극 받기 쉬운 민감한 피부가 됩니다.
    외부환경으로부터 피부를 보호하고 수분을 지키는 세라마이드의 놀라운 효능.
    일리윤의 세라마이드는 실제 피부와 가장 가까울 수 있도록 세 가지 비법을 담고 있어 더욱 특별합니다.
    """

    # Part 3: 핵심 성분 및 제품 라인업 (Ingredients & Products)
    full_text_part3 = """
    주요성분
    3가지 핵심성분
    피토 세라마이드 : 모두 똑같은 형태의 세라마이드로는 피부 장벽을 촘촘하게 채우기 어렵습니다.
    일리윤은 안정성, 효능, 피부 친화성 등 모든 면에서 탁월한 검은콩 유래 세라마이드에
    지방산 길이를 섬세하게 변화시키는 기술을 더해
    12종의 다양한 피토 세라마이드를 추출해냈습니다.
    또한 피부지질성분과 유사한 형태를 완성하기 위해 지방산과 콜레스테롤을 추가하여,
    한국인 민감 피부에서 강력한 효과를 내는 피토 세라마이드를 완성했습니다.
    세라마이드 캡슐 : 세라마이드의 효능만큼이나 중요한 것은 피부자극을 줄이는 것입니다.
    불안정한 세라마이드를 고함량으로 사용하기 위해서는
    pH가 높은 오일성분을 함께 사용하는 것이 일반적인데,
    이는 약산성 상태 유지가 중요한 민감 피부에 자극적일 수 밖에 없었습니다.
    고효능 성분과 피부 자극 사이에 고민하던 일리윤은
    세라마이드 캡슐화 기술로 두가지 과제를 한 번에 해결했습니다.
    “고함량 효능은 더 강력하게, 녹아드는 세라마이드 캡슐로 발림성은 뛰어나게.”
    까탈스러운 피부와 안목을 가진 한국인들이
    일리윤을 극찬하고 반할 수 밖에 없는 이유입니다.
    세라마이드 콤플렉스 : 한 사람의 피부에서도 부위에 따라 피부 고민과 민감도가 다릅니다.
    일리윤은 바디와 얼굴의 피부 차이를 고려하여
    각 피부에 맞는 세라마이드 콤플렉스를 개발했습니다.
    건조로 인한 가려움을 동반하는 바디 피부를 위해 고보습 케어를 설계하고,
    바디보다 얇고 노출이 잦은 얼굴 피부에는 알란토인으로 진정 효과를 더했습니다.
    또한 얼굴 피부는 보습 이후 선크림이나 파운데이션을 덧바르기에
    메이크업과의 궁합까지 고려하여 가볍고 산뜻한 제형을 적용시키는 등
    섬세한 노력을 이어가고 있습니다.
    세라마이드 아토 : 민감/건조한 피부를 위한
    세라마이드 보습 케어
    건조로 인해 가려운 피부에 강력하고 순한 보습과
    진정 효과를 전하는 저자극 장벽강화 라인입니다.
    영유아부터 성인까지, 민감피부를 위해 설계된
    특화 세라마이드 보습을 경험해보세요.
    세라마디드 더마 : 민감/건조한 피부를 위한
    세라마이드 보습 케어 (페이셜용)
    예민한 피부를 진정시키고 피부 장벽을 강화하는
    민감 페이셜 전용 더마 보습 제품으로,
    민감한 얼굴 피부를 고려한 성분과 제형으로
    촉촉하고 편안한 사용감을 경험해보세요.
    울트라 리페어: 극건조/손상 피부를 위한 판테놀 고보습 케어
    극심한 건조로 손상된 피부,
    특별한 데일리 보습이 필요합니다.
    피부에 바르는 보습 비타민 판테놀이 5% 고함량 함유된
    고밀착 고보습 울트라 리페어.
    촘촘히 차오른 극강 보습으로
    당김 없는 편안함을 경험해보세요.
    프로바이오틱스 : 자극받은 연약한 피부 수분 진정 스킨 케어
    피부에 자극을 주는 다양한 외부 환경으로 인해,
    연약해진 피부의 장벽을 튼튼하게 강화해주고
    수분 진정 시켜주는 페이셜 스킨 케어 제품과
    여성 청결제 라인입니다.
    특허 개발 프로바이오틱스 기술과 성분으로
    연약한 피부 진정을 경험해보세요.
    튼살 케어: 특별하고 깊은 피부고민에 대한
    저자극 기능성 케어일상생활의 불편함, 고민의 흔적을 남기는
    특별한 피부고민에 대한 일리윤만의 순하지만 강력한
    기능성 케어 입니다.
    연약한 임산부, 변화하는 청소년 피부에도
    편안하고 마일드한 튼살 완화 효과를 경험해보세요
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