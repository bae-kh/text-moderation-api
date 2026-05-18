# AI Text Moderation API Server

[![CI](https://github.com/bae-kh/hate-speech-filtering-api/actions/workflows/ci.yml/badge.svg)](https://github.com/bae-kh/hate-speech-filtering-api/actions/workflows/ci.yml)

HuggingFace 기반 한국어 혐오/악성 표현 분류 모델을 FastAPI 서버로 서빙하기 위한 MLOps / Backend 포트폴리오 프로젝트입니다.

이 프로젝트는 실시간 게임 채팅 사전 차단보다는, **신고된 채팅/댓글/텍스트를 자동 분류하고 운영자 검토를 보조하는 moderation API**를 1차 타겟으로 합니다.

목표는 단순히 AI 모델을 호출하는 것이 아니라, 무거운 딥러닝 모델을 실제 서비스 API로 연결하기 위해 필요한 입력 검증, 모델 로딩 최적화, confidence 기반 정책, Docker 기반 실행 환경, GitHub Actions CI 자동 검증 구조를 단계적으로 설계하는 것입니다.

---

## 1. Problem

HuggingFace 모델은 로컬에서는 쉽게 실행할 수 있지만, 실제 API 서버에 연결하면 다음 문제가 발생합니다.

- 요청마다 모델을 로딩할 경우 응답 지연과 메모리 낭비 발생
- 빈 문자열이나 비정상 입력이 모델까지 전달될 경우 불필요한 연산 발생
- 긴 텍스트 입력으로 인한 모델 입력 길이 초과 및 RuntimeError 가능성
- 모델 confidence가 낮은 경우 바로 차단할지, 검토로 보낼지에 대한 정책 필요
- 테스트/CI 환경에서 모델 로딩으로 인한 병목
- 로컬 환경과 배포 환경의 차이로 인한 재현성 문제
- Docker 환경에서 PyTorch, NumPy, HuggingFace cache 관련 의존성 문제 발생 가능성

---

## 2. Target Scenario

초기에는 이 API를 실시간 게임 채팅 사전 차단 API로 가정했습니다.

하지만 실제 게임 채팅은 사용자가 입력한 메시지가 거의 즉시 다른 사용자에게 전달되어야 하므로, 매 채팅마다 CPU-only AI 모델 API를 거치는 구조는 latency 측정 없이 실사용성을 주장하기 어렵다고 판단했습니다.

CPU-only Docker 환경에서 단일 요청 latency를 측정한 결과, 짧은 입력과 긴 입력 모두 평균 약 2.1초 수준이었습니다.

따라서 1차 타겟을 다음과 같이 조정했습니다.

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

이 구조는 실시간 사전 차단보다는, 운영자가 신고 텍스트를 검토할 때 우선순위를 정하거나 자동 분류를 보조하는 데 더 적합합니다.

---

## 3. Service Policy

이 프로젝트는 단어 단위 마스킹보다는 문장 단위 moderation 정책을 사용합니다.

현재 사용하는 `smilegate-ai/kor_unsmile` 모델은 특정 단어 위치를 반환하는 token classification 모델이 아니라, 문장 전체를 분류하는 text-classification 모델입니다.

따라서 특정 욕설 단어만 `*`로 마스킹하는 방식보다는, 입력 문장 전체에 대해 `allow`, `block`, `review` 중 하나의 action을 반환하는 정책이 현재 모델 구조에 더 적합하다고 판단했습니다.

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

| Top-1 Category | Confidence | Action | Meaning |
|---|---:|---|---|
| clean | >= 0.80 | allow | 정상으로 판단 |
| clean | < 0.80 | review | 정상 판단이 애매하므로 검토 |
| non-clean | >= 0.65 | block | 유해 가능성이 높아 차단/숨김 후보 |
| non-clean | < 0.65 | review | 유해 label이지만 확신이 낮아 검토 |

이 정책은 AI 모델의 오탐으로 인해 사용자 텍스트를 바로 차단하는 위험을 줄이고, 애매한 케이스를 운영자 검토 대상으로 분리하기 위한 설계입니다.

---

## 5. Model Understanding

이 프로젝트에서는 HuggingFace의 `smilegate-ai/kor_unsmile` 모델을 사용합니다.

해당 모델은 입력 텍스트에 대해 문장 전체 단위의 category와 confidence를 반환하는 `text-classification` 모델입니다.

즉, 특정 단어의 위치나 span을 반환하는 token classification 모델이 아닙니다.

따라서 이 모델만으로 특정 욕설 단어만 `*`로 마스킹하는 정책은 적합하지 않습니다.

이 프로젝트에서는 단어 단위 마스킹 대신, 문장 단위 moderation 정책을 사용합니다.

```text
Input text
→ HuggingFace text-classification pipeline
→ top-1 category + confidence
→ service policy
→ action: allow / block / review
```

향후 단어 단위 마스킹이 필요하다면 다음과 같은 추가 모듈이 필요합니다.

```text
- 욕설 사전 기반 필터
- token classification 모델
- span detection 모델
- 별도 profanity masking module
```

---

## 6. Length Policy

타겟을 신고 텍스트/댓글 moderation API로 조정하면서 입력 길이 정책도 수정했습니다.

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

## 7. Latency Check

CPU-only Docker 환경에서 단일 요청 latency를 측정했습니다.

측정은 로컬 Docker 컨테이너에서 실행 중인 API에 대해 `scripts/latency_check.py`로 수행했습니다.

```text
Environment:
- Local Docker container
- FastAPI
- HuggingFace pipeline
- PyTorch CPU-only
```

측정 결과는 다음과 같습니다.

| Input Length | Repeat | Average Latency | Min Latency | Max Latency |
|---:|---:|---:|---:|---:|
| 13 characters | 10 | 2094.73 ms | 2038.08 ms | 2387.72 ms |
| 720 characters | 10 | 2180.80 ms | 2148.35 ms | 2217.71 ms |

해석:

```text
- CPU-only 환경에서도 모델 추론은 정상 동작한다.
- 하지만 단일 요청 평균 latency가 약 2.1초 수준이다.
- 모든 게임 채팅 메시지를 전송 전에 검사하는 실시간 사전 차단 용도로는 현재 구조가 적합하지 않다.
- 따라서 1차 타겟을 신고 텍스트/댓글 moderation 및 운영자 검토 보조 API로 조정했다.
```

실시간 사전 차단으로 확장하려면 다음 최적화가 필요합니다.

```text
- GPU inference
- 경량 모델 검토
- ONNX Runtime 변환
- worker/thread tuning
- batch inference
- p95/p99 latency 기반 부하 테스트
```

---

## 8. Features

- FastAPI 기반 REST API 서버
- `/api/v1/health` 헬스체크 엔드포인트
- `/api/v1/detect` 신고 텍스트/댓글 moderation 엔드포인트
- Pydantic 기반 request/response schema
- 빈 문자열, 공백 문자열, 과도한 길이 입력 차단
- Request ID middleware를 통한 요청 추적성 확보
- FastAPI lifespan 기반 모델 1회 로딩
- HuggingFace `smilegate-ai/kor_unsmile` 모델 연동
- `run_in_threadpool` 기반 동기 추론 격리
- confidence 기반 `allow / block / review` 정책 적용
- Docker 기반 CPU-only 컨테이너 실행 환경
- pytest 기반 API 및 model service layer 테스트
- GitHub Actions 기반 CI 자동 검증

---

## 9. Tech Stack

| Area | Tech |
|---|---|
| Language | Python |
| Web Framework | FastAPI |
| Validation | Pydantic |
| Model Serving | HuggingFace Transformers, PyTorch CPU |
| Testing | Pytest, FastAPI TestClient, MagicMock |
| Containerization | Docker |
| CI | GitHub Actions |
| Runtime | Uvicorn |

---

## 10. Architecture

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

## 11. Project Structure

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
└── latency_check.py

.github/
└── workflows/
    └── ci.yml

Dockerfile
.dockerignore
requirements.txt
README.md
```

---

## 12. Phase Summary

### Phase 1. API MVP

AI 모델을 붙이기 전에 API 서버의 기본 구조와 입출력 계약을 먼저 고정했습니다.

Implemented:

- FastAPI 앱 구성
- `/api/v1/health` 엔드포인트 추가
- `/api/v1/detect` 엔드포인트 추가
- Pydantic 기반 Request/Response Schema 정의
- 빈 문자열, 공백 문자열, 과도한 길이 입력 차단
- Request ID middleware 추가
- Mock 기반 API 테스트 작성

Key idea:

> 실제 AI 모델 없이 API 계약과 입력 검증을 먼저 고정해, 모델 연동 전에도 서버 로직을 테스트할 수 있는 구조를 만들었습니다.

---

### Phase 2. Model Serving & Moderation Policy

HuggingFace `smilegate-ai/kor_unsmile` 모델을 FastAPI 서버에 연동하고, 모델 confidence 기반 moderation 정책을 추가했습니다.

Implemented:

- `HateSpeechModel` 클래스로 모델 로딩/추론 로직 분리
- FastAPI lifespan 기반 모델 1회 로딩
- `app.state.model`을 통한 모델 인스턴스 재사용
- `run_in_threadpool` 기반 동기 추론 격리
- top-1 category + confidence 기반 정책 적용
- `allow / block / review` action 반환
- confidence threshold 기반 review 정책 추가

Key idea:

> 단순히 label이 clean이면 allow, 아니면 block하는 방식이 아니라, confidence가 낮은 애매한 케이스는 운영자 review로 보내도록 정책을 확장했습니다.

---

### Phase 3. Dockerization

로컬 환경에 의존하던 FastAPI 모델 서빙 서버를 Docker 컨테이너로 패키징했습니다.

Implemented:

- `python:3.11.9-slim-bookworm` 기반 Docker 이미지 구성
- CPU-only PyTorch wheel 직접 설치
- `numpy==1.26.4` 고정
- non-root user 적용
- HuggingFace cache 경로 설정
- `.dockerignore` 작성
- Docker build/run 및 컨테이너 API 호출 검증

Key idea:

> 로컬 환경 의존성을 제거하기 위해 Docker로 실행 환경을 캡슐화했고, ML 의존성으로 인한 이미지 크기와 빌드 안정성 문제를 layer cache, 버전 고정, CPU-only 의존성 구성으로 관리했습니다.

---

### Phase 4. CI

GitHub Actions를 사용해 테스트와 Docker build 검증을 자동화했습니다.

Implemented:

- `.github/workflows/ci.yml` 작성
- push / pull_request 이벤트 기반 CI 실행
- Python 3.11 runner 환경 구성
- PyTorch CPU-only wheel 설치
- 프로젝트 의존성 설치
- `pytest -v` 자동 실행
- Docker image build 자동 검증

Key idea:

> 코드 변경이 API contract, validation, model service layer, confidence policy, Dockerfile을 깨뜨리지 않는지 GitHub Actions에서 자동으로 검증할 수 있게 했습니다.

---

## 13. API

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
  "confidence": 0.6500,
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

## 14. Run Locally

```bash
pip install -r requirements.txt
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
Invoke-RestMethod `
  -Uri "http://localhost:8000/api/v1/detect" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"text":"신고된 댓글 예시입니다."}'
```

---

## 15. Run with Docker

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
Invoke-RestMethod `
  -Uri "http://localhost:8000/api/v1/detect" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"text":"신고된 댓글 예시입니다."}'
```

---

## 16. Test

Run tests locally:

```bash
pytest -v
```

The test suite includes:

- `/api/v1/health` 정상 응답
- `/api/v1/detect` 정상 응답
- 빈 문자열 validation
- 공백 문자열 validation
- text 필드 누락 validation
- max_length 초과 validation
- model service layer 결과 변환 로직
- confidence threshold 기반 `allow / block / review` 정책
- RuntimeError 발생 시 HTTP 500 처리

Note:

`test_model.py`는 실제 HuggingFace 모델을 다운로드하거나 로딩하는 테스트가 아닙니다.

`MagicMock`을 사용해 HuggingFace pipeline의 반환값을 대체하고, `HateSpeechModel`의 결과 변환 로직과 예외 처리만 빠르게 검증합니다.

실제 모델 로딩과 추론은 Docker 컨테이너 실행 후 `/api/v1/detect`를 호출하는 smoke test로 별도 검증합니다.

---

## 17. CI

GitHub Actions를 사용해 테스트와 Docker build 검증을 자동화했습니다.

CI는 `push` 또는 `pull_request` 이벤트가 발생하면 실행됩니다.

검증 항목은 다음과 같습니다.

- Python 3.11 환경 구성
- PyTorch CPU-only wheel 설치
- 프로젝트 의존성 설치
- `pytest -v` 실행
- Docker image build 검증

Workflow file:

```text
.github/workflows/ci.yml
```

Local equivalent:

```bash
pytest -v
docker build -t hate-speech-api .
```

현재 CI는 실제 HuggingFace 모델을 다운로드하지 않습니다.

실제 모델 로딩과 추론은 Docker run 기반 smoke test로 별도 검증합니다.

---

## 18. Requirements

Current `requirements.txt`:

```txt
fastapi==0.110.1
uvicorn==0.29.0
pydantic==2.6.4
numpy==1.26.4
transformers==4.39.3
pytest==8.1.1
httpx==0.27.0
requests==2.32.3
```

Note:

Docker와 CI 환경에서는 PyTorch CPU-only wheel을 직접 설치합니다.

따라서 `requirements.txt`에는 `torch`를 포함하지 않습니다.

---

## 19. Trouble Shooting

### 1. PyTorch CUDA dependency issue

초기에는 `requirements.txt`에 `torch==2.2.2`를 포함했습니다.

하지만 Docker 빌드 시 CUDA 관련 패키지가 대량으로 설치되며 이미지 크기와 빌드 안정성 문제가 발생했습니다.

해결을 위해 `requirements.txt`에서 torch를 제거하고, Dockerfile에서 CPU-only PyTorch wheel을 직접 설치했습니다.

---

### 2. Base image instability

`python:3.11-slim` 기반 빌드 중 pip/Python 표준 라이브러리 단계에서 비정상 오류가 발생했습니다.

해결을 위해 base image를 다음과 같이 고정했습니다.

```dockerfile
FROM python:3.11.9-slim-bookworm
```

---

### 3. NumPy runtime compatibility issue

컨테이너 실행 후 실제 추론 요청 시 다음 오류가 발생했습니다.

```text
Numpy is not available
```

PyTorch 2.2.2 CPU wheel과 최신 NumPy 계열의 호환성 문제로 판단하고, `numpy==1.26.4`로 고정하여 해결했습니다.

---

### 4. Python bytecode cache issue

Docker build 또는 container run 과정에서 다음 오류가 발생한 적이 있습니다.

```text
ValueError: bad marshal data (invalid reference)
```

이는 Python bytecode cache 또는 Docker layer 상태 문제로 판단했습니다.

완화 방법:

- `.dockerignore`에 `__pycache__/`, `*.py[cod]` 포함
- `pip install` 단계에 `--no-compile` 옵션 추가
- 필요 시 `docker build --pull --no-cache`로 재빌드

---

### 5. CI contract mismatch

API 응답 구조에 `category`, `action` 필드를 추가한 뒤, 테스트에서 사용하는 FakeModel이 예전 응답 구조를 반환해 CI가 실패했습니다.

```text
KeyError: 'category'
```

FakeModel의 반환값을 실제 API contract에 맞게 수정하여 해결했습니다.

이 경험을 통해 CI가 API contract 변경에 따른 테스트 불일치를 조기에 감지하는 안전장치로 동작함을 확인했습니다.

---

## 20. Limitations

현재 구조에는 다음 한계가 있습니다.

- CPU-only 추론 환경으로 평균 latency가 약 2.1초 수준
- 실시간 게임 채팅 사전 차단 용도로는 아직 부적합
- 실제 운영 환경에서 p95/p99 latency와 throughput 측정 필요
- GPU inference 미적용
- ONNX Runtime 최적화 미적용
- Docker run 기반 smoke test는 CI에 아직 포함하지 않음
- 클라우드 배포 미진행
- 운영자 검토 UI 미구현
- 실제 신고 데이터 기반 threshold calibration 미진행

---

## 21. Future Work

다음 작업을 진행할 예정입니다.

- 실제 신고 텍스트 샘플 기반 threshold calibration
- `allow / block / review` 정책 고도화
- Docker run 기반 smoke test 자동화
- `/api/v1/health` 자동 smoke test
- `/api/v1/detect` 샘플 요청 자동 검증
- Locust 또는 k6 기반 부하 테스트
- 평균 latency, p95, p99 latency 측정
- CPU-only 유지 가능 여부 판단
- ONNX Runtime 변환 검토
- GPU inference 검토
- 운영자 검토 큐 또는 간단한 admin UI 연동

---

## 22. Interview Summary

이 프로젝트는 단순히 HuggingFace 모델을 FastAPI에 붙인 것이 아니라, 무거운 딥러닝 모델을 실제 API 서버로 안정적으로 서빙하기 위해 API 계약, 입력 검증, 모델 로딩 최적화, confidence 기반 moderation 정책, Docker 기반 실행 환경, CI 자동 검증 파이프라인을 단계적으로 설계한 MLOps / Backend 프로젝트입니다.


## Runtime Smoke Test

Phase 5에서는 CI를 runtime smoke test까지 확장했습니다.

기본 CI는 push 또는 pull_request마다 다음을 검증합니다.

```text
pytest
→ docker build
→ docker run
→ GET /api/v1/health