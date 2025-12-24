import numpy as np 
import pandas as pd
import pickle as pkl
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

class ToneAnalyzer:
    """
    톤 분석을 담당하는 클래스입니다.
    데이터(centroids, meta)를 인스턴스 상태로 관리하여 import 시점의 부작용을 방지합니다.
    """
    def __init__(self, tone_centroids: dict, tone_meta: pd.DataFrame):
        """
        직접 데이터를 주입받아 초기화합니다.
        파일 로딩 로직과 비즈니스 로직을 분리하기 위함입니다.
        """
        self.tone_centroids = tone_centroids
        self.tone_meta = tone_meta

    @classmethod
    def from_files(cls, pkl_path: str | Path, meta_path: str | Path):
        """
        파일 경로를 받아 데이터를 로드한 후 ToneAnalyzer 인스턴스를 반환하는 팩토리 메서드입니다.
        """
        pkl_path = Path(pkl_path)
        meta_path = Path(meta_path)

        if not pkl_path.exists():
            raise FileNotFoundError(f"Centroid file not found: {pkl_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_path}")

        # 1. Centroids 로드
        with open(pkl_path, "rb") as f:
            tone_centroids = pkl.load(f)

        # 2. Metadata 로드 (Index 설정 포함)
        tone_meta = pd.read_csv(meta_path).set_index("tone_id")

        return cls(tone_centroids, tone_meta)

    def map_tone_vector_to_params(self, tone_vector: np.ndarray) -> dict:
        """
        입력된 벡터와 가장 유사한 톤을 찾아 메타데이터를 반환합니다.
        
        입력: tone_vector (np.ndarray)
        출력: dict (매핑 결과)
        """
        sims = {}

        # 코사인 유사도 계산
        # (반복문 최적화를 위해 나중에 행렬 연산으로 바꿀 수도 있음, 현재 로직 유지)
        for tone_id, centroid in self.tone_centroids.items():
            sim = cosine_similarity(
                tone_vector.reshape(1, -1),
                centroid.reshape(1, -1)
            )[0, 0]
            sims[tone_id] = float(sim)

        # 가장 유사한 tone 선정
        selected_tone = max(sims, key=sims.get)
        
        # 메타데이터 추출
        if selected_tone not in self.tone_meta.index:
             raise ValueError(f"Tone ID '{selected_tone}' not found in metadata.")

        row = self.tone_meta.loc[selected_tone]

        return {
            "tone_id": selected_tone,
            "dominant_trait": row.get("dominant_trait"),
            "proof_level": row.get("proof_level"),
            "emotion_level": row.get("emotion_level"),
            "cta_strength": row.get("cta_strength"),
            "lexicon_group": row.get("lexicon_group"),
            "ban_group": row.get("ban_group"),
            "description": row.get("description"),
            "similarity": sims[selected_tone],
        }

# ==================================================
# 실행 테스트 (Import 시에는 실행되지 않음)
# ==================================================
if __name__ == "__main__":
    # 이 블록 안의 코드는 파일을 직접 실행할 때만 작동합니다. (python this_file.py)
    # import 시에는 무시됩니다.
    
    try:
        # 테스트를 위한 기본 경로 설정
        BASE_DIR = Path(__file__).resolve().parents[2]
        DATA_DIR = BASE_DIR / "third_week" / "data_csv"
        
        PATH_CENTROIDS = DATA_DIR / "tone_vectors.pkl"
        PATH_META = DATA_DIR / "tone_metadata_extended.csv"

        print(f"Loading assets from: {DATA_DIR}")

        # 1. 인스턴스 생성 (여기서 파일 로드 발생)
        analyzer = ToneAnalyzer.from_files(PATH_CENTROIDS, PATH_META)
        
        # 2. 더미 데이터로 테스트
        dummy_vector = np.random.rand(768) # 예: embedding size
        result = analyzer.map_tone_vector_to_params(dummy_vector)
        
        print("\n✅ Test Result:")
        print(result)

    except Exception as e:
        print(f"\n❌ Error during test execution: {e}")
