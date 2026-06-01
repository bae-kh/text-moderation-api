

# AI Text Moderation API Server

[![CI](https://github.com/bae-kh/hate-speech-filtering-api/actions/workflows/ci.yml/badge.svg)](https://github.com/bae-kh/hate-speech-filtering-api/actions/workflows/ci.yml)

HuggingFace 기반 한국어 혐오/악성 표현 분류 모델을 FastAPI 서버로 서빙하기 위한 MLOps / Backend 포트폴리오 프로젝트입니다.

이 프로젝트는 단순히 AI 모델을 호출하는 API가 아니라, 무거운 딥러닝 모델을 실제 서비스 API로 연결하기 위해 필요한 API 계약, 입력 검증, 모델 로딩 최적화, confidence 기반 moderation 정책, Docker 실행 환경, CI, runtime smoke test, load test를 단계적으로 설계한 프로젝트입니다.

현재 1차 타겟은 **실시간 게임 채팅 사전 차단**이 아니라, **신고 텍스트 / 댓글 moderation 및 운영자 검토 보조 API**입니다.

실시간 채팅 필터링은 평균 latency만으로 판단하기 어렵고, p95/p99 latency, 순간 트래픽, fallback 정책, 오탐/미탐 처리까지 함께 고려해야 합니다. 따라서 현재 프로젝트에서는 신고된 채팅, 댓글, 텍스트를 자동 분류하고 운영자 검토를 보조하는 use case를 1차 범위로 설정했습니다.

---

## 1. Problem

HuggingFace 모델은 로컬 실험에서는 쉽게 실행할 수 있지만, 실제 API 서버에 연결하면 다음과 같은 엔지니어링 문제가 발생합니다.

* 요청마다 모델을 로딩할 경우 응답 지연과 메모리 낭비 발생
* 빈 문자열, 공백 문자열, 과도한 길이 입력이 모델 계층까지 전달될 가능성
* 긴 텍스트 입력으로 인한 모델 입력 길이 초과 및 RuntimeError 가능성
* 비동기 FastAPI endpoint 안에서 동기 모델 추론을 직접 실행할 경우 이벤트 루프 blocking 가능성
* 모델 confidence가 낮은 경우 allow/block/review 중 어떤 정책을 적용할지에 대한 기준 필요
* 테스트/CI 환경에서 실제 모델 로딩으로 인한 병목
* 로컬 환경과 Docker 환경의 차이로 인한 재현성 문제
* Docker 환경에서 PyTorch, NumPy, HuggingFace cache 관련 의존성 문제 발생 가능성
* Docker image가 build되더라도 실제 container가 정상 실행되는지 별도 검증 필요
* CPU-only 환경에서 latency와 load 특성을 정량적으로 확인할 필요

이 프로젝트는 이러한 문제를 단계적으로 해결하면서, AI 모델을 실제 서비스 API로 안정적으로 serving하기 위한 backend/MLOps 구조를 설계하는 것을 목표로 합니다.

---

## 2. Target Scenario

초기에는 이 API를 실시간 게임 채팅 사전 차단 API로 가정했습니다.

하지만 실시간 게임 채팅은 사용자가 입력한 메시지가 거의 즉시 다른 사용자에게 전달되어야 하며, 평균 latency뿐 아니라 p95/p99 tail latency, 순간 트래픽, 장애 시 fallback, 오탐으로 인한 사용자 경험까지 고려해야 합니다.

따라서 현재 1차 타겟을 다음과 같이 조정했습니다.

```text
Before:
- 실시간 게임 채팅 사전 차단 API

After:
- 신고 텍스트 / 댓글 moderation API
- 운영자 검토 보조 API
```

이 API는 신고된 채팅, 댓글, 텍스트를 입력받아 모델의 `category`, `confidence`를 기반으로 서비스 처리 정책인 `action`을 반환합니다.

```text
action = "allow"
→ 정상 텍스트로 판단하여 통과

action = "block"
→ 유해 가능성이 높은 텍스트로 판단하여 자동 차단 또는 숨김 후보

action = "review"
→ 모델 confidence가 애매하므로 운영자 검토 큐로 이동
```

이 구조는 실시간 사전 차단보다, 운영자가 신고 텍스트를 검토할 때 우선순위를 정하거나 자동 분류를 보조하는 데 더 적합합니다.

---

## 3. Service Policy

이 프로젝트는 단어 단위 마스킹보다는 문장 단위 moderation 정책을 사용합니다.

현재 사용하는 `smilegate-ai/kor_unsmile` 모델은 특정 단어 위치를 반환하는 token classification 모델이 아니라, 입력 문장 전체에 대해 category와 confidence를 반환하는 text-classification 모델입니다.

따라서 특정 욕설 단어만 `*`로 마스킹하는 방식보다는, 입력 문장 전체에 대해 `allow`, `block`, `review` 중 하나의 action을 반환하는 정책이 현재 모델 구조에 더 적합하다고 판단했습니다.

향후 단어 단위 마스킹이 필요하다면 다음과 같은 별도 모듈이 필요합니다.

```text
- 욕설 사전 기반 필터
- token classification 모델
- span detection 모델
- 별도 profanity masking module
```

---

## 4. Confidence Policy

초기 구현에서는 모델의 top-1 label이 `clean`이면 `allow`, `clean`이 아니면 `block`으로 처리했습니다.

하지만 모델 출력 결과를 직접 확인해보니, 단순 이진 정책은 실제 서비스 정책으로는 부족했습니다.

예를 들어 다음과 같은 결과가 나왔습니다.

```text
TEXT: 바보
clean: 0.7924
악플/욕설: 0.1651

TEXT: 바보같아
악플/욕설: 0.6500
clean: 0.3025
```

`바보`는 top-1 label이 `clean`이지만 confidence가 0.8보다 낮아 완전히 정상이라고 보기에는 애매합니다.

반면 `바보같아`는 top-1 label이 `악플/욕설`이고 confidence가 0.65 수준으로 나타났습니다.

따라서 top-1 label은 유지하되, label 종류에 따라 서로 다른 confidence threshold를 적용했습니다.

```text
clean + confidence >= 0.80
→ allow

clean + confidence < 0.80
→ review

non-clean + confidence >= 0.65
→ block

non-clean + confidence < 0.65
→ review
```

최종 정책은 다음과 같습니다.

| Top-1 Category | Confidence | Action | Meaning               |
| -------------- | ---------: | ------ | --------------------- |
| clean          |    >= 0.80 | allow  | 정상으로 판단               |
| clean          |     < 0.80 | review | 정상 판단이 애매하므로 검토       |
| non-clean      |    >= 0.65 | block  | 유해 가능성이 높아 차단/숨김 후보   |
| non-clean      |     < 0.65 | review | 유해 label이지만 확신이 낮아 검토 |

이 정책은 AI 모델의 오탐으로 인해 사용자 텍스트를 바로 차단하는 위험을 줄이고, 애매한 케이스를 운영자 검토 대상으로 분리하기 위한 설계입니다.

---

## 5. Model Output Strategy

초기 API 구현에서는 HuggingFace `pipeline("text-classification")`의 기본 top-1 결과만 사용했습니다.

하지만 `inspect_model_output.py`로 실제 label별 score를 확인해보니, `kor_unsmile` 모델은 `clean`, `악플/욕설`, `지역`, `기타 혐오` 등 여러 label score를 반환할 수 있었습니다.

따라서 API 서버에서도 inspect script와 동일하게 전체 label score를 받은 뒤, 가장 높은 score의 label을 직접 선택하도록 수정했습니다.

```text
Input text
  ↓
HuggingFace text-classification pipeline
  ↓
top_k=None, function_to_apply="sigmoid"
  ↓
all label scores
  ↓
best label selection
  ↓
confidence policy
  ↓
action: allow / block / review
```

이 방식으로 inspect script와 실제 API 서버의 추론 조건을 맞추고, Docker API에서도 동일한 문장에 대해 일관된 결과가 나오도록 개선했습니다.

---

## 6. Length Policy

타겟을 신고 텍스트/댓글 moderation API로 조정하면서 입력 길이 정책도 명확히 분리했습니다.

```text
API input limit:
- max_length = 1000 characters

Model input limit:
- max_length = 256 tokens
- truncation = True
```

두 제한은 서로 다른 계층의 방어선입니다.

Pydantic의 `max_length=1000`은 HTTP 요청으로 들어오는 원문 문자열 길이를 제한하는 API 레벨 방어선입니다.

반면 HuggingFace pipeline의 `max_length=256`은 tokenizer 이후 실제 모델에 들어가는 token 수를 제한하는 모델 레벨 방어선입니다.

```text
Client Text
   ↓
Pydantic Validation
- max_length = 1000 characters
- blank text blocked
   ↓
Tokenizer
   ↓
Model Input
- max_length = 256 tokens
- truncation = True
   ↓
HuggingFace Model
```

긴 게시글 전체 분석은 현재 1차 범위에 포함하지 않았습니다.

장문 게시글을 정확히 분석하려면 향후 chunking 또는 sliding window 방식으로 확장할 수 있습니다.

---

## 7. Architecture

```text
Reported Text / Comment
  ↓
Game or Community Server
  ↓
POST /api/v1/detect
  ↓
FastAPI Middleware
- X-Request-ID
  ↓
Router
- /api/v1/health
- /api/v1/detect
  ↓
Pydantic Schema
- DetectRequest
- DetectResponse
  ↓
Service Layer
- HateSpeechModel
  ↓
HuggingFace Pipeline
- smilegate-ai/kor_unsmile
  ↓
Confidence Policy
- clean threshold
- harmful threshold
  ↓
Response
- category
- confidence
- action: allow / block / review
  ↓
Moderation System
- allow: 정상 처리
- block: 자동 차단 또는 숨김 후보
- review: 운영자 검토 큐로 이동
```

---

## 8. Project Structure

```text
app/
├── main.py
├── api/
│   └── routes.py
├── schemas/
│   └── payload.py
└── services/
    └── model.py

tests/
├── test_api.py
└── test_model.py

scripts/
├── inspect_model_output.py
├── latency_check.py
├── run_load_tests.ps1
├── summarize_load_tests.py
└── summarize_stress_tests.py

load_tests/
├── locustfile.py
└── locustfile_stress.py

load_test_results/
├── summary.md
└── stress_summary.md

.github/
└── workflows/
    ├── ci.yml
    └── model-smoke-test.yml

Dockerfile
.dockerignore
requirements.txt
requirements-dev.txt
README.md
```

---

## 9. Phase Summary

### Phase 1. API MVP

AI 모델을 붙이기 전에 API 서버의 기본 구조와 입출력 계약을 먼저 고정했습니다.

Implemented:

* FastAPI 앱 구성
* `/api/v1/health` 엔드포인트 추가
* `/api/v1/detect` 엔드포인트 추가
* Pydantic 기반 Request/Response Schema 정의
* 빈 문자열, 공백 문자열, 과도한 길이 입력 차단
* Request ID middleware 추가
* Mock 기반 API 테스트 작성

Key idea:

> 실제 AI 모델 없이 API 계약과 입력 검증을 먼저 고정해, 모델 연동 전에도 서버 로직을 테스트할 수 있는 구조를 만들었습니다.

---

### Phase 2. Model Serving & Moderation Policy

HuggingFace `smilegate-ai/kor_unsmile` 모델을 FastAPI 서버에 연동하고, 모델 confidence 기반 moderation 정책을 추가했습니다.

Implemented:

* `HateSpeechModel` 클래스로 모델 로딩/추론 로직 분리
* FastAPI lifespan 기반 모델 1회 로딩
* `app.state.model`을 통한 모델 인스턴스 재사용
* `run_in_threadpool` 기반 동기 추론 격리
* top-1 category + confidence 기반 정책 적용
* `allow / block / review` action 반환
* confidence threshold 기반 review 정책 추가

Key idea:

> 단순히 label이 clean이면 allow, 아니면 block하는 방식이 아니라, confidence가 낮은 애매한 케이스는 운영자 review로 보내도록 정책을 확장했습니다.

---

### Phase 3. Dockerization

로컬 환경에 의존하던 FastAPI 모델 서빙 서버를 Docker 컨테이너로 패키징했습니다.

Implemented:

* `python:3.11.9-slim-bookworm` 기반 Docker 이미지 구성
* CPU-only PyTorch wheel 직접 설치
* `numpy==1.26.4` 고정
* non-root user 적용
* HuggingFace cache 경로 설정
* `.dockerignore` 작성
* Docker build/run 및 컨테이너 API 호출 검증

Key idea:

> 로컬 환경 의존성을 제거하기 위해 Docker로 실행 환경을 캡슐화했고, ML 의존성으로 인한 이미지 크기와 빌드 안정성 문제를 layer cache, 버전 고정, CPU-only 의존성 구성으로 관리했습니다.

---

### Phase 4. CI

GitHub Actions를 사용해 테스트와 Docker build 검증을 자동화했습니다.

Implemented:

* `.github/workflows/ci.yml` 작성
* push / pull_request 이벤트 기반 CI 실행
* Python 3.11 runner 환경 구성
* PyTorch CPU-only wheel 설치
* 프로젝트 의존성 설치
* `pytest -v` 자동 실행
* Docker image build 자동 검증
* CI badge 추가

Key idea:

> 코드 변경이 API contract, validation, model service layer, confidence policy, Dockerfile을 깨뜨리지 않는지 GitHub Actions에서 자동으로 검증할 수 있게 했습니다.

---

### Phase 5. Runtime Smoke Test

Phase 5에서는 CI를 build 검증에서 runtime 검증으로 확장했습니다.

기존 Phase 4에서는 Docker image가 정상적으로 build되는지만 검증했습니다. 하지만 Docker image가 만들어진다고 해서 실제 컨테이너가 정상적으로 실행되고, FastAPI 서버가 살아있는지는 보장되지 않습니다.

따라서 Phase 5에서는 GitHub Actions에서 Docker container를 실제로 실행한 뒤 `/api/v1/health` endpoint를 호출하는 runtime smoke test를 추가했습니다.

Implemented:

* GitHub Actions에서 Docker image build
* Docker container background 실행
* `/api/v1/health` endpoint 호출
* health response body 검증
* 실패 시 `docker logs` 출력
* 성공/실패와 관계없이 container cleanup 수행
* 실제 모델 추론 smoke test는 `workflow_dispatch` 기반 manual workflow로 분리

Key idea:

> Docker build 성공 여부뿐 아니라, build된 image가 실제 runtime에서 FastAPI server로 정상 기동되는지까지 검증했습니다.

CI에서 수행하는 기본 smoke test 흐름은 다음과 같습니다.

```text
git push / pull_request
  ↓
pytest
  ↓
docker build
  ↓
docker run
  ↓
GET /api/v1/health
  ↓
200 OK + {"status": "ok"} 확인
```

실제 `/api/v1/detect` 모델 추론 smoke test는 HuggingFace 모델 로딩과 CPU inference 비용이 크기 때문에 매 push마다 실행하지 않고, 필요할 때 수동으로 실행할 수 있는 workflow로 분리했습니다.

```text
Manual Model Smoke Test
  ↓
docker build
  ↓
docker run
  ↓
GET /api/v1/health
  ↓
POST /api/v1/detect
  ↓
response schema 확인
```

---

### Phase 6. Latency & Load Test

Phase 6에서는 CPU-only HuggingFace 모델 서버의 latency와 load 특성을 정량적으로 확인했습니다.

초기 측정에서는 약 2초대의 latency가 관측된 적이 있었지만, 이후 cold start / warm 상태 / 측정 조건을 분리해 다시 확인했습니다.

최종 성능 측정은 FastAPI lifespan에서 모델이 이미 메모리에 로드된 **warm 상태**를 기준으로 수행했습니다.

Implemented:

* `latency_check.py`로 warm 상태 단일 요청 latency 측정
* `requirements-dev.txt`로 개발/부하 테스트 의존성 분리
* Locust 기반 scenario load test 작성
* Locust 기반 stress test 작성
* `scripts/run_load_tests.ps1`로 users 1/5/10/20 scenario 자동 실행
* `scripts/summarize_load_tests.py`로 CSV 결과를 Markdown summary로 변환
* `scripts/summarize_stress_tests.py`로 stress test 결과를 Markdown summary로 변환
* Avg latency, p95, p99, RPS, Failure Rate 기록

Key idea:

> 평균 latency만 보는 것이 아니라 p95/p99 tail latency, RPS, failure rate를 함께 측정해 CPU-only inference 서버의 성능 특성을 분석했습니다.

---

## Phase 7. Moderation Review Queue

Phase 7에서는 기존 `/api/v1/detect` API가 `allow`, `block`, `review` action을 반환하고 끝나는 구조에서, 실제 운영자가 검토할 수 있는 Review Queue 구조로 확장했습니다.

기존 구조에서는 모델이 `block` 또는 `review`로 판단해도 해당 결과가 저장되지 않았습니다. 따라서 운영자가 어떤 텍스트를 검토해야 하는지, 어떤 요청이 차단되었는지, 오탐 여부를 어떻게 기록할지 알 수 없었습니다.

Phase 7에서는 `block` 또는 `review` 결과를 DB에 저장하고, 운영자가 해당 record를 조회한 뒤 최종 판단을 업데이트할 수 있도록 API를 추가했습니다.

### Motivation

기존 `/detect` API는 다음과 같은 흐름이었습니다.

```text
Text Input
  ↓
POST /api/v1/detect
  ↓
HuggingFace Model Inference
  ↓
Confidence Policy
  ↓
allow / block / review
  ↓
Response 반환
```

하지만 실제 moderation system에서는 `review`나 `block` 결과가 반환된 뒤에도 추가 workflow가 필요합니다.

```text
review가 나오면 운영자가 어디서 확인하는가?
block된 기록은 어디에 남는가?
오탐이면 어떻게 수정하는가?
운영자 판단 결과를 나중에 threshold 조정이나 모델 개선에 활용할 수 있는가?
```

이를 해결하기 위해 Phase 7에서는 Review Queue를 추가했습니다.

### Review Queue Flow

```text
POST /api/v1/detect
  ↓
AI Model Inference
  ↓
Confidence Policy
  ↓
action 결정
  ├── allow  → 저장하지 않음
  ├── block  → DB 저장
  └── review → DB 저장
  ↓
moderation_records
  ↓
GET /api/v1/moderation/records
  ↓
운영자 검토
  ↓
PATCH /api/v1/moderation/records/{record_id}
  ↓
status = resolved
```

### Storage Policy

Phase 7에서는 모든 요청을 저장하지 않고, 운영적으로 의미가 있는 `block`과 `review` 결과만 저장합니다.

| Action | DB 저장 여부 | 이유                                     |
| ------ | -------- | -------------------------------------- |
| allow  | 저장하지 않음  | 정상으로 판단된 요청이므로 review queue에 쌓을 필요가 없음 |
| block  | 저장       | 자동 차단 또는 숨김 후보로 운영자 확인 가능              |
| review | 저장       | 모델 confidence가 애매하므로 운영자 검토 필요         |

`allow` 요청까지 모두 저장하면 DB가 불필요하게 커지고, 운영자가 확인할 필요 없는 데이터가 review queue에 쌓일 수 있습니다. 따라서 MVP 단계에서는 `block`과 `review`만 저장하도록 설계했습니다.

### Database

Phase 7에서는 SQLite와 SQLAlchemy를 사용했습니다.

```text
SQLite
- 파일 기반 DB
- 별도 DB 서버가 필요 없음
- 로컬 개발과 MVP 검증에 적합

SQLAlchemy
- Python에서 DB를 다루기 위한 ORM 라이브러리
- DB table을 Python class로 정의
- SQLite에서 PostgreSQL로 확장하기 쉬운 구조 제공
```

현재는 MVP 단계이므로 `moderation.db`라는 SQLite 파일에 데이터를 저장합니다.

실제 운영 환경으로 확장한다면 PostgreSQL과 Docker Compose 기반으로 DB를 분리할 계획입니다.

### moderation_records Table

Phase 7에서 추가한 핵심 테이블은 `moderation_records`입니다.

| Field          | Meaning              |
| -------------- | -------------------- |
| id             | record 고유 ID         |
| text           | 검토 대상 텍스트            |
| is_hate_speech | 모델의 유해 표현 판단 여부      |
| category       | 모델이 선택한 top category |
| confidence     | 모델 confidence score  |
| action         | block 또는 review      |
| status         | pending 또는 resolved  |
| review_result  | 운영자의 최종 판단           |
| review_note    | 운영자 검토 메모            |
| created_at     | 생성 시간                |
| updated_at     | 수정 시간                |

### Status and Review Result

`status`와 `review_result`는 서로 다른 의미를 가집니다.

```text
status
- 검토 진행 상태
- pending: 아직 운영자가 검토하지 않음
- resolved: 운영자가 검토 완료

review_result
- 운영자의 최종 판단 결과
- confirmed_harmful: 실제 유해 표현으로 확인
- false_positive: 모델 오탐으로 판단
- clean: 정상 표현으로 판단
```

예를 들어 처음 저장된 record는 다음과 같습니다.

```json
{
  "status": "pending",
  "review_result": null,
  "review_note": null
}
```

운영자가 검토를 완료하면 다음과 같이 변경됩니다.

```json
{
  "status": "resolved",
  "review_result": "confirmed_harmful",
  "review_note": "운영자 검토 결과 유해 표현으로 판단됨"
}
```

### Added APIs

Phase 7에서 다음 API를 추가했습니다.

#### Get moderation records

```http
GET /api/v1/moderation/records
```

저장된 moderation record 목록을 조회합니다.

Query parameter를 통해 `status`, `action`, `limit` 기준으로 필터링할 수 있습니다.

Example:

```http
GET /api/v1/moderation/records?status=pending&limit=20
```

#### Get moderation record detail

```http
GET /api/v1/moderation/records/{record_id}
```

특정 moderation record의 상세 정보를 조회합니다.

#### Update moderation review

```http
PATCH /api/v1/moderation/records/{record_id}
```

운영자의 최종 검토 결과를 저장합니다.

Request:

```json
{
  "review_result": "confirmed_harmful",
  "review_note": "운영자 검토 결과 유해 표현으로 판단됨"
}
```

Response:

```json
{
  "id": 1,
  "status": "resolved",
  "review_result": "confirmed_harmful",
  "review_note": "운영자 검토 결과 유해 표현으로 판단됨"
}
```

PATCH를 사용한 이유는 기존 record 전체를 교체하는 것이 아니라, `status`, `review_result`, `review_note` 같은 일부 필드만 수정하기 때문입니다.

### Test Automation

Phase 7에서는 Review Queue workflow를 pytest로 자동화했습니다.

추가한 테스트는 다음과 같습니다.

| Test                   | Purpose                          |
| ---------------------- | -------------------------------- |
| allow 결과는 저장하지 않음      | 정상 텍스트가 review queue에 쌓이지 않는지 검증 |
| block 결과는 pending으로 저장 | 유해 텍스트가 DB에 저장되는지 검증             |
| record 상세 조회           | 특정 moderation record 조회 검증       |
| PATCH 후 resolved 처리    | 운영자 검토 완료 workflow 검증            |
| 존재하지 않는 record 조회      | 404 응답 검증                        |

테스트에서는 실제 HuggingFace 모델을 호출하지 않고 `FakeModerationModel`을 사용했습니다. 테스트 목적은 모델 성능 검증이 아니라, 모델이 `allow`, `block`, `review` 결과를 반환했을 때 DB workflow가 올바르게 동작하는지 검증하는 것이기 때문입니다.

또한 실제 `moderation.db` 파일을 사용하지 않고 `sqlite:///:memory:` 기반 인메모리 SQLite DB를 사용했습니다. 이를 통해 테스트가 실제 로컬 DB 상태에 영향을 받지 않고, 매번 깨끗한 DB에서 독립적으로 실행되도록 했습니다.

Test result:

```text
17 passed
```

### Key Result

Phase 7을 통해 기존 AI model serving API는 다음과 같이 확장되었습니다.

```text
Before:
AI 모델 추론 결과 반환

After:
AI 모델 추론 결과 반환
+ block/review 결과 저장
+ 운영자 검토 목록 조회
+ 운영자 최종 판단 업데이트
+ 테스트 자동화
```

이를 통해 단순 모델 서빙 API에서 운영 가능한 AI moderation backend로 한 단계 확장했습니다.
---



## 10. API

### Health Check

```http
GET /api/v1/health
```

Response:

```json
{
  "status": "ok",
  "message": "API server is running"
}
```

---

### Detect Text

```http
POST /api/v1/detect
Content-Type: application/json
```

Request:

```json
{
  "text": "신고된 댓글 예시입니다."
}
```

Allow response:

```json
{
  "is_hate_speech": false,
  "confidence": 0.9287,
  "category": "clean",
  "action": "allow",
  "message": "Message allowed."
}
```

Block response:

```json
{
  "is_hate_speech": true,
  "confidence": 0.9136,
  "category": "악플/욕설",
  "action": "block",
  "message": "Message blocked due to harmful content."
}
```

Review response:

```json
{
  "is_hate_speech": false,
  "confidence": 0.7924,
  "category": "clean",
  "action": "review",
  "message": "Message requires human review."
}
```

Validation error:

```json
{
  "text": ""
}
```

Response:

```http
422 Unprocessable Entity
```

---

## 11. Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run server:

```bash
uvicorn app.main:app --reload
```

Swagger UI:

```text
http://localhost:8000/docs
```

Health check:

```bash
curl http://localhost:8000/api/v1/health
```

Detect:

```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
  -H "Content-Type: application/json" \
  -d '{"text":"신고된 댓글 예시입니다."}'
```

PowerShell:

```powershell
$body = @{
  text = "신고된 댓글 예시입니다."
} | ConvertTo-Json -Compress

Invoke-RestMethod `
  -Uri "http://localhost:8000/api/v1/detect" `
  -Method POST `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
```

---

## 12. Run with Docker

Build image:

```bash
docker build -t hate-speech-api .
```

Run container:

```bash
docker run --rm -p 8000:8000 hate-speech-api
```

Swagger UI:

```text
http://localhost:8000/docs
```

PowerShell test:

```powershell
$body = @{
  text = "신고된 댓글 예시입니다."
} | ConvertTo-Json -Compress

Invoke-RestMethod `
  -Uri "http://localhost:8000/api/v1/detect" `
  -Method POST `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
```

---

## 13. Test

Run tests locally:

```bash
pytest -v
```

The test suite includes:

* `/api/v1/health` 정상 응답
* `/api/v1/detect` 정상 응답
* 빈 문자열 validation
* 공백 문자열 validation
* text 필드 누락 validation
* max_length 초과 validation
* model service layer 결과 변환 로직
* confidence threshold 기반 `allow / block / review` 정책
* multi-label pipeline output 처리
* RuntimeError 발생 시 HTTP 500 처리

Note:

`test_model.py`는 실제 HuggingFace 모델을 다운로드하거나 로딩하는 테스트가 아닙니다.

`MagicMock`을 사용해 HuggingFace pipeline의 반환값을 대체하고, `HateSpeechModel`의 결과 변환 로직과 예외 처리만 빠르게 검증합니다.

실제 모델 로딩과 추론은 Docker 컨테이너 실행 후 `/api/v1/detect`를 호출하는 smoke test로 별도 검증합니다.

---

## 14. CI

GitHub Actions를 사용해 테스트와 Docker build 검증을 자동화했습니다.

CI는 `push` 또는 `pull_request` 이벤트가 발생하면 실행됩니다.

검증 항목은 다음과 같습니다.

* Python 3.11 환경 구성
* PyTorch CPU-only wheel 설치
* 프로젝트 의존성 설치
* `pytest -v` 실행
* Docker image build
* Docker container run
* `/api/v1/health` runtime smoke test
* 실패 시 Docker logs 출력
* container cleanup

Workflow file:

```text
.github/workflows/ci.yml
```

Local equivalent:

```bash
pytest -v
docker build -t hate-speech-api .
docker run --rm -p 8000:8000 hate-speech-api
```

---

## 15. Runtime Smoke Test

Phase 5에서는 Docker image가 build되는 것에서 끝나지 않고, 실제 container가 정상 실행되는지 검증하는 runtime smoke test를 추가했습니다.

Smoke test는 전체 기능을 깊게 검증하는 테스트가 아니라, 배포 또는 실행 직후 시스템이 최소한 정상 기동되는지 확인하는 최소 런타임 테스트입니다.

CI에서는 다음을 자동으로 확인합니다.

```text
docker build
→ docker run
→ GET /api/v1/health
→ 200 OK 확인
```

이를 통해 Dockerfile이 build되는지만 확인하는 것이 아니라, build된 image가 실제로 FastAPI 서버로 기동되는지까지 검증할 수 있습니다.

실제 모델 추론은 HuggingFace 모델 다운로드와 CPU inference 비용이 크기 때문에 매 push마다 실행하지 않고, `workflow_dispatch` 기반 수동 model smoke test로 분리했습니다.

Manual model smoke test는 다음을 검증합니다.

```text
docker build
→ docker run
→ GET /api/v1/health
→ POST /api/v1/detect
→ response schema 확인
```

이렇게 기본 CI와 실제 모델 추론 검증을 분리함으로써, 빠르고 안정적인 기본 CI와 필요 시 실행 가능한 실제 inference 검증을 모두 확보했습니다.

---

## 16. Latency Test

Latency 측정은 FastAPI lifespan에서 모델이 이미 메모리에 로드된 **warm 상태**를 기준으로 수행했습니다.

Warm 상태란 서버가 이미 기동되었고, HuggingFace pipeline과 model weight가 메모리에 올라가 있어 요청이 들어오면 바로 추론 가능한 상태를 의미합니다.

Cold start 또는 모델 다운로드/초기화 시간은 이번 latency 측정에 포함하지 않았습니다.

### Single Request Latency

|   Input Length | Repeat | Avg Latency (ms) | Min Latency (ms) | Max Latency (ms) |
| -------------: | -----: | ---------------: | ---------------: | ---------------: |
|  13 characters |     10 |            30.05 |            16.49 |            53.60 |
| 720 characters |     10 |            87.19 |            77.83 |            96.63 |

해석:

* 짧은 입력은 평균 약 30ms로 처리되었습니다.
* 긴 입력은 평균 약 87ms로 처리되었습니다.
* 입력 길이가 증가하면 tokenizer/model input 처리 비용이 증가해 latency도 증가했습니다.
* 이 결과는 모델이 이미 메모리에 로드된 warm 상태 기준입니다.
* cold start latency는 별도 지표로 분리해 측정해야 합니다.

---

## 17. Load Test

Phase 6에서는 Locust를 사용해 `/api/v1/detect` endpoint의 성능을 측정했습니다.

`/api/v1/health`는 서버 생존 여부를 확인하는 endpoint이므로 load test 대상이 아닙니다.

실제 모델 추론이 발생하는 `/api/v1/detect`를 대상으로 부하 테스트를 수행했습니다.

---

### Development Requirements

부하 테스트 도구는 운영 서버 실행에 필요한 의존성이 아니므로, `requirements-dev.txt`로 분리했습니다.

```txt
-r requirements.txt
locust==2.32.6
```

Install:

```bash
pip install -r requirements-dev.txt
```

---

### Scenario Load Test

Scenario test는 `wait_time = between(1, 3)`을 적용했습니다.

즉, 각 가상 사용자가 요청을 보낸 뒤 1~3초 대기하도록 구성해 실제 신고/댓글 moderation 요청 흐름에 가까운 부하를 만들었습니다.

이 테스트는 서버의 최대 처리량을 압박하는 stress test가 아니라, 일반적인 사용 흐름에서 latency와 안정성을 확인하기 위한 테스트입니다.

Test condition:

```text
Target:
POST /api/v1/detect

Input Mix:
- 75% short reported comments
- 25% long reported comments

Wait Time:
- 1~3 seconds between requests

Users:
- 1
- 5
- 10
- 20

Metrics:
- Average latency
- p95 latency
- p99 latency
- RPS
- Failure rate
```

Scenario load test result:

| Users | Avg Latency (ms) | p95 Latency (ms) | p99 Latency (ms) |  RPS | Failure Rate (%) |
| ----: | ---------------: | ---------------: | ---------------: | ---: | ---------------: |
|     1 |            74.96 |              130 |              140 | 0.49 |             0.00 |
|     5 |            90.38 |              160 |              170 | 2.30 |             0.00 |
|    10 |            86.48 |              150 |              170 | 4.56 |             0.00 |
|    20 |            88.31 |              170 |              200 | 8.87 |             0.00 |

Interpretation:

* 동시 사용자 20명까지 평균 latency는 약 90ms 이하로 유지되었습니다.
* p99 latency도 200ms 이하로 측정되었습니다.
* 모든 scenario에서 failure rate는 0%였습니다.
* RPS는 사용자 수 증가에 따라 증가했습니다.
* 이 결과는 wait_time이 포함된 scenario test 기준이며, 서버의 최대 처리량을 의미하지는 않습니다.

---

### Stress Test

Stress test는 `wait_time` 없이 가능한 빠르게 `/api/v1/detect` 요청을 보내도록 구성했습니다.

이 테스트는 실제 사용자 흐름보다는 CPU-only 모델 서버의 한계 지점을 확인하기 위한 목적입니다.

Stress test를 수행하는 이유는 다음과 같습니다.

```text
- 서버가 어디서부터 버거워지는지 확인
- CPU-only inference 병목 확인
- p95/p99 latency가 언제 증가하는지 확인
- RPS가 어느 지점에서 더 이상 증가하지 않는지 확인
- failure가 발생하는지 확인
```

Stress test result:

| Users | Avg Latency (ms) | p95 Latency (ms) | p99 Latency (ms) |   RPS | Failure Rate (%) |
| ----: | ---------------: | ---------------: | ---------------: | ----: | ---------------: |
|     5 |            77.22 |            82.00 |            89.00 | 62.43 |             0.00 |
|    10 |           190.85 |           220.00 |           240.00 | 48.27 |             0.00 |
|    20 |           398.98 |           500.00 |           550.00 | 45.98 |             0.00 |

Interpretation:

* 5 users에서는 평균 latency 77.22ms, p99 89ms로 안정적이었습니다.
* 10 users부터 평균 latency가 190.85ms로 증가했고, RPS는 48.27로 감소했습니다.
* 20 users에서는 평균 latency가 398.98ms, p99가 550ms까지 증가했습니다.
* failure rate는 모든 stress scenario에서 0%였습니다.
* 사용자 수가 5명에서 20명으로 증가하면서 latency는 증가했지만, RPS는 10 users 이후 더 이상 증가하지 않고 감소했습니다.
* 이는 CPU-only inference 서버가 특정 지점 이후 처리량 확장보다 queue 대기와 latency 증가로 반응한다는 것을 보여줍니다.
* 현재 CPU-only baseline은 제한된 동시성에서는 안정적으로 동작하지만, 대규모 실시간 트래픽을 처리하려면 GPU inference, ONNX Runtime, worker tuning, batch inference 등의 최적화가 필요합니다.

---

## 18. Testing Strategy

이 프로젝트의 테스트는 계층별로 나누어 설계했습니다.

| Test Type          | Scope                                  | Tool                | Runs On               |
| ------------------ | -------------------------------------- | ------------------- | --------------------- |
| Unit Test          | model service logic, confidence policy | pytest, MagicMock   | local, CI             |
| API Contract Test  | request/response schema, validation    | pytest, TestClient  | local, CI             |
| Docker Build Test  | image build 가능 여부                      | docker build        | CI                    |
| Runtime Smoke Test | container run, health endpoint         | docker run, curl    | CI                    |
| Model Smoke Test   | actual model loading and inference     | docker run, /detect | manual workflow/local |
| Scenario Load Test | realistic request interval load        | Locust              | local/manual          |
| Stress Test        | no-wait high-pressure load             | Locust              | local/manual          |

핵심 구분은 다음과 같습니다.

```text
pytest:
코드 로직과 API contract 검증

docker build:
이미지 생성 가능 여부 검증

runtime smoke test:
build된 이미지가 실제 container로 실행되고 server가 살아나는지 검증

model smoke test:
실제 HuggingFace 모델 로딩과 /detect 추론 경로 검증

scenario load test:
실제 신고/댓글 moderation 요청 흐름에 가까운 부하 검증

stress test:
서버 한계와 CPU-only inference 병목 확인
```

---

## 19. Troubleshooting

### 1. PyTorch CUDA dependency issue

초기에는 `requirements.txt`에 `torch==2.2.2`를 포함했습니다.

하지만 Docker 빌드 시 CUDA 관련 패키지가 대량으로 설치되며 이미지 크기와 빌드 안정성 문제가 발생했습니다.

해결을 위해 `requirements.txt`에서 torch를 제거하고, Dockerfile에서 CPU-only PyTorch wheel을 직접 설치했습니다.

---

### 2. NumPy runtime compatibility issue

컨테이너 실행 후 실제 추론 요청 시 다음 오류가 발생했습니다.

```text
Numpy is not available
```

PyTorch 2.2.2 CPU wheel과 최신 NumPy 계열의 호환성 문제로 판단하고, `numpy==1.26.4`로 고정하여 해결했습니다.

---

### 3. Python bytecode cache issue

Docker build 또는 container run 과정에서 다음 오류가 발생한 적이 있습니다.

```text
ValueError: bad marshal data (invalid reference)
```

완화 방법:

* `.dockerignore`에 `__pycache__/`, `*.py[cod]` 포함
* `pip install` 단계에 `--no-compile` 옵션 추가
* 필요 시 `docker build --pull --no-cache`로 재빌드

---

### 4. API contract mismatch

API 응답 구조에 `category`, `action` 필드를 추가한 뒤, 테스트에서 사용하는 FakeModel이 예전 응답 구조를 반환해 CI가 실패했습니다.

```text
KeyError: 'category'
```

FakeModel의 반환값을 실제 API contract에 맞게 수정하여 해결했습니다.

이 경험을 통해 CI가 API contract 변경에 따른 테스트 불일치를 조기에 감지하는 안전장치로 동작함을 확인했습니다.

---

### 5. Docker API and inspect script mismatch

`inspect_model_output.py`에서는 특정 문장이 `악플/욕설`로 잘 분류되었지만, Docker API에서는 같은 문장이 `clean`으로 반환되는 문제가 있었습니다.

원인은 inspect script와 API 서버의 HuggingFace pipeline 설정이 달랐기 때문입니다.

해결을 위해 API 서버의 pipeline도 `top_k=None`, `function_to_apply="sigmoid"` 기반으로 전체 label score를 받은 뒤, 가장 높은 score의 label을 직접 선택하도록 수정했습니다.

이를 통해 inspect script와 실제 API 서버의 추론 조건을 일치시켰습니다.

---

## 20. Limitations

현재 구조에는 다음 한계가 있습니다.

* CPU-only inference 기반
* cold start latency는 별도 측정 필요
* 실시간 게임 채팅 사전 차단 용도로 사용하려면 더 높은 동시성과 p95/p99 검증 필요
* 대규모 트래픽 대응을 위한 worker/thread tuning 미진행
* GPU inference 미적용
* ONNX Runtime 최적화 미적용
* batch inference 미적용
* Docker run 기반 `/detect` model smoke test는 매 push CI에 포함하지 않음
* 클라우드 배포 미진행
* 운영자 검토 UI 미구현
* review/block 결과 저장 미구현
* 실제 신고 데이터 기반 threshold calibration 미진행

---

## 21. Future Work

다음 작업을 진행할 예정입니다.

* review/block 결과 저장
* 운영자 검토 queue API 추가
* 실제 신고 텍스트 샘플 기반 threshold calibration
* `allow / block / review` 정책 고도화
* structured logging 추가
* request_id, latency_ms, category, confidence, action 로그화
* Docker run 기반 model smoke test 수동 workflow 고도화
* `/api/v1/detect` 샘플 요청 자동 검증 범위 확대
* cold start latency 별도 측정
* worker/thread tuning 실험
* ONNX Runtime 변환 검토
* GPU inference 검토
* batch inference 검토
* 클라우드 VM 또는 container registry 기반 배포

---

## 22. Interview Summary

이 프로젝트는 단순히 HuggingFace 모델을 FastAPI에 붙인 것이 아니라, 무거운 딥러닝 모델을 실제 API 서버로 안정적으로 서빙하기 위해 API 계약, 입력 검증, 모델 로딩 최적화, confidence 기반 moderation 정책, Docker 실행 환경, CI 자동 검증, runtime smoke test, load test를 단계적으로 설계한 MLOps / Backend 프로젝트입니다.

특히 FastAPI lifespan을 통해 모델을 1회만 로딩하고, `run_in_threadpool`을 사용해 동기 HuggingFace pipeline 추론이 이벤트 루프를 직접 block하지 않도록 분리했습니다.

또한 Docker 기반 runtime smoke test를 통해 image가 실제 container로 기동되는지 검증했고, Locust 기반 scenario/stress test를 통해 CPU-only inference 서버의 latency, p95/p99, RPS, failure rate를 정량적으로 확인했습니다.

현재 1차 타겟은 실시간 게임 채팅 사전 차단이 아니라 신고 텍스트/댓글 moderation 및 운영자 검토 보조 API입니다. 다만 warm 상태 latency와 scenario load test 결과를 바탕으로, 향후 GPU/ONNX/worker tuning/fallback policy를 추가하면 준실시간 필터링으로 확장할 가능성도 검토할 수 있습니다.
