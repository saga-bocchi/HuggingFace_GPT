# 🧠 HuggingFace GPT-2 생성 제어 실험 (temperature, top_k, top_p)

## 📌 목표
- 텍스트 생성 모델이 다양한 설정값에 따라 얼마나 다르게 반응하는지 실험
- `temperature`, `top_k`, `top_p` 파라미터의 역할 이해 및 비교 분석

---

## 💻 실습 코드

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
prompt = input("프롬프트를 입력하세요: ")

settings = [
    {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    {"temperature": 1.0, "top_k": 50, "top_p": 0.95},
    {"temperature": 1.2, "top_k": 50, "top_p": 0.95},
    {"temperature": 0.7, "top_k": 10, "top_p": 0.9},
    {"temperature": 1.0, "top_k": 100, "top_p": 0.8},
]

for i, setting in enumerate(settings):
    print(f"\n=== 결과 {i+1} | Temp: {setting['temperature']} | Top-k: {setting['top_k']} | Top-p: {setting['top_p']} ===")
    output = generator(prompt, max_length=50, num_return_sequences=1, **setting)
    print(output[0]['generated_text'])

| 파라미터          | 설명                                                        |
| ------------- | --------------------------------------------------------- |
| `temperature` | 확률 분포의 "날카로움" 조절. 낮으면 보수적, 높으면 창의적. (보통 0.7 \~ 1.2 사이 사용) |
| `top_k`       | 확률이 높은 상위 k개 토큰만 샘플링 대상으로 사용                              |
| `top_p`       | 누적 확률 p 이하인 토큰들만 고려 (nucleus sampling)                    |

