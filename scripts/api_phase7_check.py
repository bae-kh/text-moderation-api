import requests


BASE_URL = "http://localhost:8000"


def print_response(title: str, response: requests.Response) -> None:
    print("=" * 80)
    print(title)
    print("STATUS:", response.status_code)
    print(response.json())


def main() -> None:
    # 1. detect - block/review 저장 대상 생성
    detect_response = requests.post(
        f"{BASE_URL}/api/v1/detect",
        json={"text": "바보같아"},
        timeout=30,
    )
    print_response("POST /api/v1/detect", detect_response)

    # 2. records 목록 조회
    records_response = requests.get(
        f"{BASE_URL}/api/v1/moderation/records",
        timeout=30,
    )
    print_response("GET /api/v1/moderation/records", records_response)

    records = records_response.json()
    if not records:
        print("No moderation records found.")
        return

    record_id = records[0]["id"]

    # 3. 운영자 판단 업데이트
    update_response = requests.patch(
        f"{BASE_URL}/api/v1/moderation/records/{record_id}",
        json={
            "review_result": "confirmed_harmful",
            "review_note": "운영자 검토 결과 유해 표현으로 판단됨",
        },
        timeout=30,
    )
    print_response(f"PATCH /api/v1/moderation/records/{record_id}", update_response)

    # 4. 상세 조회
    detail_response = requests.get(
        f"{BASE_URL}/api/v1/moderation/records/{record_id}",
        timeout=30,
    )
    print_response(f"GET /api/v1/moderation/records/{record_id}", detail_response)


if __name__ == "__main__":
    main()
