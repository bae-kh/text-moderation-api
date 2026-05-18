from locust import HttpUser, between, task


class ModerationApiUser(HttpUser):
    """
    AI moderation API를 호출하는 가상의 사용자 시나리오입니다.

    wait_time:
    각 사용자가 요청을 보낸 뒤 다음 요청까지 기다리는 시간입니다.
    너무 0에 가깝게 두면 실제 사용자라기보다 순수 압박 테스트가 됩니다.
    """
    wait_time = between(1, 3)

    @task(3)
    def detect_short_reported_comment(self) -> None:
        payload = {
            "text": "신고된 댓글 예시입니다."
        }

        with self.client.post(
            "/api/v1/detect",
            json=payload,
            name="POST /api/v1/detect short",
            catch_response=True,
            timeout=30,
        ) as response:
            self._validate_detect_response(response)

    @task(1)
    def detect_long_reported_comment(self) -> None:
        payload = {
            "text": "이 문장은 신고된 댓글의 길이가 긴 경우를 가정한 테스트입니다. " * 20
        }

        with self.client.post(
            "/api/v1/detect",
            json=payload,
            name="POST /api/v1/detect long",
            catch_response=True,
            timeout=30,
        ) as response:
            self._validate_detect_response(response)

    def _validate_detect_response(self, response) -> None:
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