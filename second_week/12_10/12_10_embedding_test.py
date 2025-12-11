import requests
import json

def embed(text: str):
    # Ollama embedding API 호출
    r = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )

    print("========== RAW RESPONSE START ==========")
    print(r.text)
    print("========== RAW RESPONSE END ==========")

    data = r.json()
    return data.get("embedding")

if __name__ == "__main__":
    vector = embed("이 문장을 임베딩합니다.")
    print("임베딩 길이:", len(vector) if vector else None)
    print("앞 5개:", vector[:5] if vector else None)