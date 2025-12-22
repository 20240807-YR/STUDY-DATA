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

# --- 실행부 (데이터 원문 100% 반영: 미쟝센) ---
if __name__ == "__main__":
    
    # ==========================================
    # 1. 설정: 브랜드 이름 변경
    # ==========================================
    current_brand_name = "미쟝센"

    # 2. 데이터 입력 (보내주신 텍스트 전체 포함)
    
    # Part 1: 브랜드 비전 (SHINE YOUR SCENE)
    full_text_part1 = """
    SHINE YOUR SCENE
    글로벌 헤어패션 브랜드 미쟝센
    미쟝센은 대한민국 No.1 헤어 브랜드이자, 스타일리시한 글로벌 헤어 패션 브랜드를 지향합니다.
    무대 위 아티스트의 자신감 있는 모습처럼, 미쟝센은 혁신적인 솔루션과 트렌디한 감각으로 빛나는 스타일을 완성합니다.
    개성 있는 나만의 헤어를 통해 자유로운 자기표현과 대담한 변화를 응원하며, 누구나 자신만의 무대에서 당당히 빛날 수 있도록 함께합니다.
    Shine Your Scene, Mise-en-Scène
    """

    # Part 2: 브랜드 철학 및 유래 (Style Director)
    full_text_part2 = """
    CHANGE YOUR HAIR, CHANGE YOUR LIFE.
    당신의 스타일 디렉터,
    미쟝센
    Mise-en-Scène은 '화면 속을 배치하다'라는 뜻의 프랑스어에서 유래되었습니다.
    특별히 현대 연극과 영화에서는 인물부터 분장, 조명, 배경, 카메라의 움직임까지
    세심하게 고려한 미학적 연출을 의미합니다.
    미쟝센은 당신의 헤어스타일과 이미지를 완성도 높게 디렉팅한다는
    철학을 지닌 페셔너블 헤어토털 브랜드입니다.
    """

    # Part 3: 고객 경험 및 가치 (일상이라는 무대)
    full_text_part3 = """
    미쟝센을 만나는 순간,
    모든 곳이 무대
    미쟝센에게 당신의 일상 공간은 무대, 하루의 순간들은 영화의 한 컷과 같습니다.
    시대의 트렌드, 뷰티, 라이프스타일이 정교하게 함축되어 강렬한 이미지를 남기듯,
    미장센은 스타일의 시작이자 완성이라 할 수 있는 헤어스타일로
    당신의 일상을 빛낼 것입니다.
    건강한 머릿결을 위한 프리미엄한 터치와 케어는 물론,
    당신이 원하는 스타일로의 완벽한 연출을 위해 준비된
    미쟝센의 혁신적인 제품과 전문적인 솔루션, 스타일리시한 컨텐츠.
    이제 당신의 일상은 보다 모던하고 트렌디하며 세련된 무대가 됩니다.
    미쟝센과 함께하는 순간,
    당신은 일상이라는
    무대의 주인공
    당신에게 헤어스타일은 어떤 의미인가요?
    미쟝센은 단순히 세팅을 넘어 당신의 삶을 바꿀 수 있는 시작점이라고 생각합니다.
    영화 속 주인공이 배역에 따라 스타일을 바꿔 변신하듯,
    두려움과 망설임 앞에 시도하지 않았던 스타일을 즐기도록,
    하나하나 당신만의 스타일을 만들어가도록
    용기와 변화를 주는 브랜드가 될 것입니다.
    완벽하게 갖춰진 스타일은 당신의 표정과 자세에도 자신감을 주며,
    개성있고 세련된 스타일과 라이프스타일로 당당하게 타인의
    시선을 사로잡도록 할 것입니다.
    준비 됐나요?
    자, 이제 미쟝센과 함께 일상이라는 무대의 주인공이 될 시간입니다.
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