from locust import HttpUser, task


class StressModerationApiUser(HttpUser):
    """
    wait_time 없이 가능한 빠르게 /api/v1/detect 요청을 보내는 stress test입니다.

    목적:
    - 서버의 최대 처리량 압박
    - CPU-only inference 병목 확인
    - p95/p99 latency 증가 지점 확인
    """

    @task
    def detect_reported_comment(self) -> None:
        payload = {
            "text": "신고된 댓글 예시입니다."
        }

        with self.client.post(
            "/api/v1/detect",
            json=payload,
            name="POST /api/v1/detect stress",
            catch_response=True,
            timeout=30,
        ) as response:
            if response.status_code != 200:
                response.failure(f"Unexpected status code: {response.status_code}")
                return

            try:
                data = response.json()
            except ValueError:
                response.failure("Response is not valid JSON")
                return

            required_fields = [
                "is_hate_speech",
                "confidence",
                "category",
                "action",
                "message",
            ]

            missing_fields = [
                field for field in required_fields
                if field not in data
            ]

            if missing_fields:
                response.failure(f"Missing fields: {missing_fields}")
                return

            if data["action"] not in ["allow", "block", "review"]:
                response.failure(f"Unexpected action: {data['action']}")
                return

            response.success()
