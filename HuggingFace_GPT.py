from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

user_prompt = input("프롬프트를 입력하세요: ")

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
