## 올라마는.. 파싱이 안 돼서..
# lmstudio_wrapper.py
import requests
import json

LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL = "Llama-3.1-8B-Instruct-Q4_K_M"   # LM Studio에서 다운받은 모델 이름 적기


def ask_lmstudio(prompt):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }

    r = requests.post(LMSTUDIO_URL, json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def parse_ingredients(text: str):
    prompt = f"""
    아래 전성분 문자열을 JSON 배열로 파싱하라.
    전성분:
    {text}
    반드시 ["성분1", "성분2"] 형태의 JSON array만 출력하라.
    """

    raw = ask_lmstudio(prompt)

    try:
        return json.loads(raw)
    except:
        return []