import glob
import pandas as pd
import os
from kiwipiepy import Kiwi  # [변경] Okt 대신 Kiwi 불러오기
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# ==========================================
# [설정] 파일 경로 및 브랜드 톤 사전
# ==========================================

# 현재 실행 중인 파일의 위치를 가져옵니다
current_dir = os.path.dirname(os.path.abspath(__file__))

# 상위 폴더(..)로 나간 뒤 'brand_identity_txt' 폴더를 찾습니다
FILE_PATH_PATTERN = os.path.join(current_dir, '..', 'brand_identity_txt', '*.txt')
FILE_PATH_PATTERN = os.path.abspath(FILE_PATH_PATTERN)

TONE_DICT = {
    '전문적/과학적 (Scientific)': [
        '연구', '데이터', '기술', '특허', '임상', '메커니즘', '효능', '성분', '솔루션', '혁신', '분석', '검증'
    ],
    '감성적/따뜻한 (Emotional)': [
        '사랑', '행복', '마음', '위로', '함께', '추억', '소중한', '느낌', '감동', '선물', '일상', '여유'
    ],
    '고급/프리미엄 (Luxury)': [
        '프리미엄', '고품격', '가치', '특별한', '최고', '럭셔리', '노블레스', '장인', '헤리티지', '압도적'
    ],
    '캐주얼/친근한 (Casual)': [
        '진짜', '대박', '완전', '가성비', '꿀팁', '그냥', '솔직', '친구', '쉬운', '간편', '추천'
    ]
}

# ==========================================
# 1. 데이터 로드
# ==========================================
print(f"📂 [{FILE_PATH_PATTERN}] 경로의 파일들을 불러옵니다...")

file_paths = glob.glob(FILE_PATH_PATTERN)
all_texts = []

for path in file_paths:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            all_texts.append(f.read())
    except Exception as e:
        print(f"Error reading {path}: {e}")

if not all_texts:
    print("❌ 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

df = pd.DataFrame({'text': all_texts})
print(f"✅ 총 {len(df)}개의 파일을 성공적으로 로드했습니다.\n")

kiwi = Kiwi()

# ==========================================
# 2. TF-IDF 핵심 키워드 추출 (Kiwi 적용)
# ==========================================
print("🔍 1. TF-IDF 키워드 분석 중...")

# [변경] Kiwi를 이용한 명사 추출 함수
def noun_tokenizer(text):
    tokens = kiwi.tokenize(text)
    # 태그가 N으로 시작하는 것(NNG, NNP 등 명사)만 추출
    return [t.form for t in tokens if t.tag.startswith('N')]

tfidf_vect = TfidfVectorizer(tokenizer=noun_tokenizer, min_df=2, max_df=0.90)
tfidf_matrix = tfidf_vect.fit_transform(df['text'])

feature_names = tfidf_vect.get_feature_names_out()
sum_tfidf = tfidf_matrix.sum(axis=0)

keyword_rank = []
for col, term in enumerate(feature_names):
    keyword_rank.append((term, sum_tfidf[0, col]))

keyword_df = pd.DataFrame(keyword_rank, columns=['Keyword', 'Score']).sort_values('Score', ascending=False)

print("\n[📊 Top 10 핵심 키워드]")
print(keyword_df.head(10).to_string(index=False))
print("-" * 30)


# ==========================================
# 3. 형태소 기반 말투/표현 분석 (Kiwi 적용)
# ==========================================
print("\n🔍 2. 말투(어미) 패턴 분석 중...")

ending_list = []
total_nouns = []

for text in df['text']:
    # [변경] Kiwi 토큰화
    tokens = kiwi.tokenize(text)
    
    # 태그가 E로 시작하면 어미(Ending), N으로 시작하면 명사(Noun)
    endings = [t.form for t in tokens if t.tag.startswith('E')]
    nouns = [t.form for t in tokens if t.tag.startswith('N')]
    
    ending_list.extend(endings)
    total_nouns.extend(nouns)

ending_counts = Counter(ending_list).most_common(10)

print("\n[🗣️ 자주 사용되는 어미 Top 10]")
for word, count in ending_counts:
    print(f"- {word}: {count}회")
print("-" * 30)


# ==========================================
# 4. 감성 경향 및 브랜드 톤 스코어링
# ==========================================
print("\n🔍 3. 브랜드 톤(Tone & Manner) 측정 중...")

tone_scores = {tone: 0 for tone in TONE_DICT.keys()}
total_noun_counter = Counter(total_nouns)

for tone, keywords in TONE_DICT.items():
    for keyword in keywords:
        if keyword in total_noun_counter:
            tone_scores[tone] += total_noun_counter[keyword]

print("\n[🎨 브랜드 톤 분석 결과]")
sorted_scores = sorted(tone_scores.items(), key=lambda x: x[1], reverse=True)

for tone, score in sorted_scores:
    print(f"- {tone}: {score}점")

dominant_tone = sorted_scores[0][0]
print(f"\n👉 결론: 이 브랜드는 **'{dominant_tone}'** 성향이 가장 강합니다.")