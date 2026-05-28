import requests

url = "http://localhost:8000/api/v1/detect"

texts = [
    "테스트 문장입니다.",
    "바보같아",
    "야 이 씹새끼야 뭐하냐 나가 죽어라",
]

for text in texts:
    response = requests.post(
        url,
        json={"text": text},
        timeout=30,
    )

    print("=" * 80)
    print("TEXT:", text)
    print("STATUS:", response.status_code)
    print("RESPONSE:", response.json())
