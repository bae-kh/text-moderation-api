# AI Text Moderation Backend (`text-moderation-api`)

[![CI](https://github.com/bae-kh/text-moderation-api/actions/workflows/ci.yml/badge.svg)](https://github.com/bae-kh/text-moderation-api/actions/workflows/ci.yml)

한국어 유해 표현 분류 모델(`smilegate-ai/kor_unsmile`)을 FastAPI 기반 API로 서빙하고, confidence threshold policy에 따라 **allow / block / review** action을 결정하는 AI Model Serving Backend 프로젝트입니다.

block/review 결과는 PostgreSQL에 저장되며, 운영자는 Review Queue API를 통해 모델 판단 결과를 조회하고 검토 상태를 수정할 수 있습니다.

> 이 프로젝트는 엄격한 의미의 완전한 MLOps 플랫폼은 아닙니다.
> 학습된 AI 모델을 FastAPI API, PostgreSQL review workflow, Docker Compose, structured logging, CI, performance test, threshold calibration과 연결한 **AI Model Serving Backend + Review Workflow** 포트폴리오입니다.

---

## 1. Project Overview

### Problem

AI 모델은 단순히 추론 코드로 존재하는 것만으로는 서비스가 되기 어렵습니다.
실제 백엔드 서비스로 사용하려면 API contract, 입력 검증, 모델 로딩 방식, DB 저장, 운영자 검토 workflow, 로그, 테스트, 성능 측정, threshold 정책 관리가 함께 필요합니다.

### Goal

이 프로젝트의 목표는 한국어 유해 표현 분류 모델을 단순 추론 코드가 아니라, 다음 요소를 갖춘 백엔드 서비스 형태로 확장하는 것입니다.

- FastAPI 기반 model serving API
- allow / block / review action policy
- PostgreSQL 기반 review queue
- X-API-Key 기반 관리자 API 보호
- Docker Compose 기반 API + DB 실행 환경
- structured JSON logging
- GitHub Actions CI / runtime smoke test
- latency / load / stress test
- threshold calibration report

---

## 2. Key Features

| Area | Description |
|---|---|
| Model Serving | HuggingFace `smilegate-ai/kor_unsmile` 모델을 FastAPI API로 서빙 |
| API Contract | Pydantic 기반 request/response schema 및 입력 검증 |
| Model Lifecycle | FastAPI lifespan에서 모델 1회 로딩 후 `app.state.model`로 재사용 |
| Async Handling | 동기 HuggingFace pipeline 추론을 `run_in_threadpool`로 분리 |
| Action Policy | confidence threshold 기반 `allow` / `block` / `review` 분기 |
| Review Workflow | block/review 결과를 PostgreSQL에 저장하고 운영자 검토 API 제공 |
| Admin Auth | X-API-Key 기반 관리자 API 보호, 401/403 구분 |
| DB / Migration | SQLAlchemy + PostgreSQL + Alembic migration |
| Runtime | Docker Compose 기반 API + PostgreSQL 2-container 구성 |
| Logging | `request_id`, `latency_ms`, `action`, `text_length` 중심 JSON structured logging |
| CI / Test | GitHub Actions, pytest, Docker build, runtime smoke test |
| Performance | Locust 기반 scenario load test / no-wait stress test |
| Calibration | 60건 pilot dataset으로 56개 threshold 조합 비교 |

---

## 3. Architecture

```text
Reported Text / Comment
  ↓
POST /api/v1/detect
  ↓
FastAPI Middleware
  - X-Request-ID 생성
  - request latency 측정
  ↓
Pydantic Validation
  - 필드 누락 검증
  - 빈 문자열 / 공백 문자열 차단
  - 길이 제한 검증
  ↓
HateSpeechModel
  - smilegate-ai/kor_unsmile
  - category / confidence 계산
  ↓
Confidence Policy
  ├── clean + confidence >= 0.80      → allow  → response only
  ├── clean + confidence < 0.80       → review → DB 저장
  ├── non-clean + confidence >= 0.65  → block  → DB 저장
  └── non-clean + confidence < 0.65   → review → DB 저장
  ↓
PostgreSQL moderation_records
  - block / review 결과 저장
  - allow는 저장하지 않음
  ↓
Admin Moderation API
  - X-API-Key 인증
  - list / detail / patch review result
```

Admin API:

```text
GET   /api/v1/moderation/records
GET   /api/v1/moderation/records/{record_id}
PATCH /api/v1/moderation/records/{record_id}
```

---

## 4. Quick Start

### 4.1 Clone

```bash
git clone https://github.com/bae-kh/text-moderation-api.git
cd text-moderation-api
```

### 4.2 Run with Docker Compose

```bash
docker compose up --build
```

첫 실행 시 HuggingFace 모델 다운로드로 시간이 걸릴 수 있습니다.
이후 실행부터는 `hf-cache` volume으로 모델 파일이 캐시됩니다.

Swagger UI:

```text
http://localhost:8000/docs
```

### 4.3 Stop

```bash
docker compose down
```

DB 데이터까지 초기화하려면:

```bash
docker compose down -v
```

---

## 5. Basic API Check

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

Expected response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "db_connected": true
}
```

### Detect Text

```bash
curl -X POST http://localhost:8000/api/v1/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "오늘 날씨가 정말 좋네요"}'
```

Example response:

```json
{
  "is_hate_speech": false,
  "confidence": 0.98,
  "category": "clean",
  "action": "allow",
  "message": "allowed"
}
```

> README에서는 유해 표현 원문 노출을 피하기 위해 정상 텍스트 예시만 제공합니다.
> block/review 흐름은 테스트 코드, calibration report, 내부 실행 캡처로 검증했습니다.

### Review Queue

```bash
curl http://localhost:8000/api/v1/moderation/records \
  -H "X-API-Key: docker-admin-key-change-me"
```

---

## 6. API Reference

### Health Check

```http
GET /api/v1/health
```

| Field | Description |
|---|---|
| `status` | API server status |
| `model_loaded` | 모델 로딩 여부 |
| `db_connected` | DB 연결 여부 |

---

### Detect

```http
POST /api/v1/detect
Content-Type: application/json
```

Request:

```json
{
  "text": "검사할 텍스트"
}
```

Response fields:

| Field | Description |
|---|---|
| `is_hate_speech` | 모델의 유해 표현 판단 여부 |
| `confidence` | top category confidence score |
| `category` | 모델이 선택한 top category |
| `action` | `allow` / `block` / `review` |
| `message` | 처리 결과 메시지 |

---

### Moderation Records

관리자 API는 `X-API-Key` header가 필요합니다.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/moderation/records` | 검토 목록 조회, pagination/filter 지원 |
| GET | `/api/v1/moderation/records/{record_id}` | 단일 record 상세 조회 |
| PATCH | `/api/v1/moderation/records/{record_id}` | 운영자 검토 결과 저장 |

PATCH request example:

```json
{
  "review_result": "confirmed_harmful",
  "review_note": "운영자 검토 결과 유해 표현으로 판단"
}
```

Auth error:

| Status | Meaning |
|---|---|
| 401 | X-API-Key header 없음 |
| 403 | X-API-Key 값이 틀림 |

---

## 7. Confidence Policy

| Top-1 Category | Confidence | Action | Meaning |
|---|---:|---|---|
| clean | >= 0.80 | allow | 정상으로 판단 |
| clean | < 0.80 | review | 정상 판단이지만 confidence가 낮아 검토 |
| non-clean | >= 0.65 | block | 유해 가능성이 높아 자동 차단 후보 |
| non-clean | < 0.65 | review | 유해 label이지만 confidence가 낮아 검토 |

threshold 값은 `pydantic-settings` 기반 환경변수로 주입할 수 있습니다.

현재 값(`clean=0.80`, `harmful=0.65`)은 60건 pilot calibration 결과를 바탕으로 해석한 보수적 baseline 정책입니다.

---

## 8. Project Structure

```text
app/
├── main.py                   # FastAPI app + lifespan + middleware
├── api/
│   ├── routes.py             # /api/v1/health, /api/v1/detect
│   └── moderation.py         # /api/v1/moderation/records
├── core/
│   ├── config.py             # pydantic-settings 기반 설정
│   ├── exceptions.py         # domain exception
│   ├── logging.py            # structured logging
│   └── security.py           # X-API-Key auth
├── db/
│   ├── database.py           # DB connection
│   └── models.py             # SQLAlchemy models
├── schemas/
│   ├── payload.py            # detect request/response schema
│   └── moderation.py         # moderation record schema
└── services/
    ├── model.py              # HateSpeechModel
    └── moderation_store.py   # DB CRUD

tests/                        # pytest tests
scripts/                      # latency_check, calibration scripts
load_tests/                   # Locust load/stress test
load_test_results/            # performance test results
calibration_results/          # threshold calibration report + raw data
.github/workflows/            # CI + model smoke test
```

---

## 9. Implementation Phase Summary

이 프로젝트는 API MVP에서 threshold calibration까지 단계적으로 확장했습니다.

| Phase | Topic | Key Point |
|---|---|---|
| 1 | API MVP | 모델 연동 전 API contract와 입력 검증을 먼저 고정 |
| 2 | Model Serving | lifespan 기반 1회 로딩 + `run_in_threadpool` 분리 |
| 3 | Dockerization | CPU-only PyTorch + non-root user + `.dockerignore` |
| 4 | CI | GitHub Actions로 pytest + Docker build 자동 검증 |
| 5 | Runtime Smoke Test | container 실행 후 `/api/v1/health` 호출 검증 |
| 6 | Performance Test | Locust로 latency / RPS / failure 측정 |
| 7 | Review Queue | block/review 결과를 DB에 저장하고 운영자 검토 workflow 구성 |
| 8 | Structured Logging | request_id, latency_ms, action 중심 JSON log, 원문 미포함 |
| 8.5 | Config & Code Quality | 설정 외부화 + custom exception으로 계층 책임 분리 |
| 9 | PostgreSQL + Compose | SQLite MVP에서 PostgreSQL 기반 API+DB 구조로 확장 |
| 10 | Auth + Pagination | X-API-Key 인증, 401/403 구분, limit/offset + filter |
| 11 | Alembic Migration | create_all 의존 축소, migration 기반 schema 변경 관리 |
| 12 | Threshold Calibration | 60건 pilot dataset으로 56개 threshold 조합 비교 |

상세 Phase 기록:

```text
working/02_project_phase_summary.md
```

---

## 10. Performance Summary

### Test Environment

| Item | Value |
|---|---|
| Purpose | 운영 서버 기준 절대 성능이 아니라, 제한된 CPU-only 환경에서 동시 요청 증가에 따른 latency/RPS 변화 확인 |
| Environment | Local machine |
| CPU | 13th Gen Intel(R) Core(TM) i9-13900K |
| RAM | 32GB |
| GPU | 사용하지 않음, CPU-only inference |
| Run mode | local uvicorn |
| Model | `smilegate-ai/kor_unsmile` |
| Tools | `latency_check.py`, Locust |

### Warm Latency Test

| Test | Avg Latency |
|---|---:|
| Short text | 약 30ms |
| Long text | 약 87ms |

### Scenario Load Test

`wait_time=1~2s` 조건에서 실제 사용 패턴에 가까운 요청 간격을 둔 테스트입니다.

| Users | Avg Latency (ms) | p95 (ms) | p99 (ms) | RPS | Failure |
|---:|---:|---:|---:|---:|---:|
| 1 | 73.89 | 130 | 130 | 0.51 | 0.00% |
| 5 | 91.96 | 150 | 150 | 2.41 | 0.00% |
| 10 | 89.60 | 150 | 170 | 4.46 | 0.00% |
| 20 | 92.91 | 180 | 200 | 8.99 | 0.00% |

### Stress Test

`no wait` 조건에서 가능한 한 빠르게 요청을 보내는 압박 테스트입니다.

| Users | Avg Latency (ms) | p95 (ms) | p99 (ms) | RPS | Failure |
|---:|---:|---:|---:|---:|---:|
| 5 | 77.22 | 82 | 89 | 62.43 | 0.00% |
| 10 | 190.85 | 220 | 240 | 48.27 | 0.00% |
| 20 | 398.98 | 500 | 550 | 45.98 | 0.00% |

### Interpretation

Scenario test에서는 20 users까지 평균 latency 약 93ms, p99 약 200ms, failure 0%로 비교적 안정적인 응답을 보였습니다.

반면 no-wait stress test에서는 동시 요청 증가에 따라 평균 latency와 p99 latency가 증가하고, RPS가 일정 수준에서 정체되는 것을 확인했습니다. 이는 현재 CPU-only 단일 API 구조에서 모델 추론이 병목으로 작용할 수 있음을 보여줍니다.

---

## 11. Pivot Decision

초기 목표는 **실시간 게임 채팅 사전 차단 API**였습니다.

그러나 CPU-only stress test 결과, 동시 요청이 증가할수록 latency가 증가하고 RPS가 정체되는 것을 확인했습니다. 실시간 채팅 사전 차단은 평균 latency뿐 아니라 p95/p99 tail latency, 순간 트래픽, fallback 정책까지 고려해야 합니다.

따라서 현재 구조에서는 대규모 실시간 사전 차단 경로에 직접 넣기보다, **신고 텍스트·댓글 moderation을 보조하고 block/review 결과를 운영자가 검토하는 review queue 기반 moderation backend**가 더 현실적인 1차 타겟이라고 판단했습니다.

이 판단은 실시간 moderation 자체가 불가능하다는 의미가 아닙니다. 향후 ONNX Runtime, quantization, caching, batching, GPU inference 등을 적용한 뒤 실시간 차단 가능성을 다시 평가할 수 있습니다.

---

## 12. Threshold Calibration

60건 pilot calibration dataset(clean 30, harmful 30)을 사용해 현재 threshold 정책과 56개 threshold 조합을 비교했습니다.

### Current Threshold Performance

현재 설정:

```text
clean_allow_threshold = 0.80
harmful_block_threshold = 0.65
```

| Metric | Value |
|---|---:|
| Auto-block Precision | 100.0% |
| Auto-block Recall | 56.7% |
| Auto-block F1 | 72.3% |
| Safety Coverage | 73.3% |
| Harmful Allow Rate | 26.7% |
| Review Rate | 10.0% |

### Threshold Selection Criteria

```text
1순위: 정상 텍스트 자동 차단 위험을 보수적으로 관리하기
2순위: review rate를 과도하게 높이지 않기
3순위: pilot 결과만으로 aggressive block 정책을 바로 기본값으로 확정하지 않기
4순위: 실제 review_result 데이터 기반으로 threshold 재조정하기
```

현재 threshold는 최적값이 아니라, 정상 텍스트 자동 차단 위험을 보수적으로 관리하면서 review queue 기반 운영 흐름을 관찰하기 위한 baseline 정책입니다.

상세 calibration report:

```text
calibration_results/calibration_report.md
```

---

## 13. Testing Strategy

| Test Type | Scope | Tool | Environment |
|---|---|---|---|
| Unit Test | model service logic, confidence policy | pytest + MagicMock | local, CI |
| API Contract Test | request/response schema, validation | pytest + TestClient | local, CI |
| Auth / Pagination / Filter | 인증, pagination, category filter | pytest + TestClient | local, CI |
| Docker Build | image build 가능 여부 | docker build | CI |
| Runtime Smoke Test | container 기동 + `/api/v1/health` 응답 | docker run + curl | CI |
| Model Smoke Test | 실제 모델 추론 | docker run + `/api/v1/detect` | manual workflow |
| Scenario Load Test | 실제 사용 패턴 기반 부하 | Locust | local |
| Stress Test | 최대 동시 요청 압박 | Locust | local |

---

## 14. Structured Logging

`/api/v1/detect` 요청마다 JSON structured log를 기록합니다.

```json
{
  "event": "detect_completed",
  "request_id": "7b740556-41ce-4b2c-9201-d9d3a77e203e",
  "method": "POST",
  "path": "/api/v1/detect",
  "status_code": 200,
  "latency_ms": 693.17,
  "text_length": 4,
  "category": "악플/욕설",
  "confidence": 0.7818,
  "action": "block",
  "stored": true
}
```

Logging policy:

- raw text는 로그에 저장하지 않습니다.
- 로그에는 `text_length`, `category`, `confidence`, `action`, `latency_ms` 등 분석에 필요한 최소 정보만 기록합니다.
- DB에는 운영자 review를 위해 원문을 저장하되, 외부 유출 가능성이 있는 로그에서는 제외했습니다.
- `request_id`를 통해 요청 단위 추적이 가능합니다.

---

## 15. Agentic AI Usage

Agentic AI는 개발 속도를 높이기 위한 Pair Programming 파트너로 활용했습니다.

활용 범위:

- 반복적인 boilerplate 코드 초안 작성 보조
- 테스트 케이스 아이디어 도출
- Docker/CI 설정 초안 작성 보조
- 문서 구조화
- 디버깅 방향 탐색
- 성능 테스트 및 threshold 결과 정리 보조

직접 판단한 부분:

- 프로젝트 주제 선정
- allow / block / review action policy
- review queue 구조
- raw text를 로그에 남기지 않는 logging policy
- 실시간 차단에서 review queue 기반 moderation backend로 피벗한 판단
- calibration 결과를 보수적 baseline 정책으로 해석한 판단

AI가 제안한 코드 초안은 검증 없이 반영하지 않았고, pytest, Docker Compose 실행, API 호출, docker logs 확인, DB 저장 확인, calibration 결과 대조를 통해 직접 검증했습니다.

---

## 16. Run Locally

Docker Compose 사용을 권장합니다.

로컬 개발 환경에서 실행하려면:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Swagger UI:

```text
http://localhost:8000/docs
```

---

## 17. Test

```bash
pip install -r requirements-dev.txt
pytest -v
```

`test_model.py`는 `MagicMock`으로 HuggingFace pipeline을 대체합니다.
실제 모델 추론 검증은 Docker 기반 smoke test 또는 manual model smoke workflow로 수행합니다.

---

## 18. Limitations

- 본 프로젝트는 완전한 MLOps 플랫폼이 아닙니다.
- 모델 학습 파이프라인, model registry, drift monitoring, continuous training은 구현하지 않았습니다.
- 현재 성능 테스트는 제한된 로컬 CPU-only 환경에서 수행되었습니다.
- 실시간 게임 채팅 사전 차단 용도로는 추가 최적화가 필요합니다.
- X-API-Key 인증은 MVP 수준의 관리자 API 보호 방식입니다.
- Threshold calibration은 60건 pilot dataset 기준이며, 실제 운영 데이터 기반 재측정이 필요합니다.
- 운영자 검토 UI는 구현하지 않았고, API만 제공합니다.
- 클라우드 배포는 진행하지 않았습니다.
- worker/thread tuning과 cold start latency 측정은 향후 과제로 남겼습니다.

---

## 19. Future Work

- 운영 데이터 기반 threshold 재측정
- category별 세분화된 threshold 검토
- ONNX Runtime 변환
- dynamic quantization 적용 검토
- Redis 기반 빈출 텍스트 캐싱
- batch inference 구조 검토
- GPU inference 또는 별도 inference service 분리
- worker/thread tuning
- Prometheus/Grafana 기반 latency/error monitoring
- 클라우드 배포
- model registry 또는 모델 버전 관리 도입

---

## 20. Troubleshooting

<details>
<summary>PyTorch CUDA dependency issue</summary>

일반 `torch` dependency를 설치하면 불필요한 CUDA 관련 패키지가 설치될 수 있습니다.
Dockerfile에서 CPU-only PyTorch wheel을 직접 설치하여 해결했습니다.

</details>

<details>
<summary>NumPy runtime compatibility issue</summary>

PyTorch CPU wheel과 최신 NumPy의 호환성 문제로 `numpy==1.26.4`로 고정했습니다.

</details>

<details>
<summary>Docker API와 inspect script의 추론 결과 불일치</summary>

inspect script와 API 서버의 HuggingFace pipeline 설정이 달라 confidence 결과가 다르게 나왔습니다.
API 서버에도 동일한 `softmax` 기반 설정을 적용하여 추론 조건을 일치시켰습니다.

</details>

<details>
<summary>PowerShell에서 한국어 JSON 요청 인코딩 문제</summary>

PowerShell에서 curl로 한국어 JSON을 보내면 인코딩 문제가 발생할 수 있습니다.
`Invoke-RestMethod`와 UTF-8 명시적 사용으로 해결했습니다.

```powershell
$body = [System.Text.Encoding]::UTF8.GetBytes('{"text": "검사할 텍스트"}')
Invoke-RestMethod -Uri http://localhost:8000/api/v1/detect `
  -Method POST -Body $body -ContentType "application/json; charset=utf-8"
```

</details>
