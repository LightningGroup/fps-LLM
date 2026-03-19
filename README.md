# LangGraph 기반 LLM 오케스트레이션 프로젝트

Python `LangGraph`와 `FastAPI`로 구성한 LLM 오케스트레이션.

## 기술 스택

- Python 3.11+
- LangGraph
- FastAPI
- Uvicorn
- Pydantic

## 실행 방법

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

## API 엔드포인트

- `GET /health`: 헬스체크
- `POST /threads`: 새 `thread_id` 발급
- `POST /chat`: 대화 실행(필요 시 승인 인터럽트 반환)
- `POST /chat/approval`: 승인/거절 값으로 인터럽트 지점 재개
- `GET /v1/models`: 호환 모델 목록
- `POST /v1/chat/completions`: Chat Completions 호환 API
- `POST /v1/responses`: Responses 호환 API

## 요청 예시

```bash
# 1) thread_id 발급
curl -s http://localhost:8000/threads -X POST

# 2) 일반 질의
curl -s http://localhost:8000/chat -X POST \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"<THREAD_ID>","message":"환불 정책 알려줘"}'

# 3) 액션 질의(승인 필요)
curl -s http://localhost:8000/chat -X POST \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"<THREAD_ID>","message":"고객에게 환불 처리 메일 보내줘"}'

# 4) 인터럽트 재개
curl -s http://localhost:8000/chat/approval -X POST \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"<THREAD_ID>","decision":"approved"}'

# 5) Chat Completions 호환 호출
curl -s http://localhost:8000/v1/chat/completions -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-key" \
  -d '{
    "model":"corp-gpt",
    "messages":[
      {"role":"developer","content":"답변을 간결하게 해줘"},
      {"role":"user","content":"환불 정책 알려줘"}
    ]
  }'

# 6) Responses 호환 호출
curl -s http://localhost:8000/v1/responses -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-key" \
  -d '{
    "model":"corp-gpt",
    "instructions":"핵심만 답변해줘",
    "input":"환불 정책 알려줘"
  }'
```

## 프로젝트 구조

```text
app/
  __init__.py
  api.py                  # FastAPI 엔트리포인트
  graph.py                # LangGraph 조립
  state.py                # State 정의
  routing.py              # 요청 분기/플래닝 로직
  retrieval.py            # Retrieval 노드
  action.py               # Action 노드 + interrupt
  tools.py                # 외부 도구 시뮬레이션
  vector_store.py         # 데모 벡터DB
```
