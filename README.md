# LangGraph 기반 LLM 오케스트레이션 프로젝트

요청하신 아키텍처(Direct / Retrieval / Action + HITL + Checkpoint)를 Python `LangGraph`로 구성한 예제 프로젝트입니다.

## 핵심 특징

- `Ingest -> Route -> Direct/Retrieval/Action` 분기
- Retrieval Workflow
  - `Plan Retrieval -> Retrieve Docs -> Grade Docs -> Rewrite Query(루프) -> Generate Answer -> Answer Check`
- Action Workflow
  - `Plan Action -> approval_interrupt(interrupt) -> Execute Tool -> Observe Result -> Finalize Answer`
- `thread_id` + `MemorySaver` checkpointer를 통한 상태 저장/재개
- 인메모리 벡터 스토어(데모용)와 도구 실행 시뮬레이터 포함

## 실행 방법

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m app.main
```


## FastAPI 서버 실행

CLI 대신 HTTP API로 실행할 수 있습니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

주요 엔드포인트:

- `GET /health`: 헬스체크
- `POST /threads`: 새 `thread_id` 발급
- `POST /chat`: 대화 실행(필요 시 승인 인터럽트 반환)
- `POST /chat/approval`: 승인/거절 값으로 인터럽트 지점 재개

예시 요청:

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
```

## 시작점(Entry Point)과 실행 흐름

- **실행 시작점**: `app/main.py` 의 `main()` 함수 (`python -m app.main`으로 호출됨)
- `main()` 내부에서 `build_app()`를 호출해 LangGraph app을 생성
- 사용자 입력을 `app.invoke(...)`로 전달하면 그래프가 `START -> ingest -> route` 순서로 실행
- `route` 판단에 따라 `direct/retrieval/action` 경로로 분기
- `action` 경로에서 `approval_interrupt`가 발생하면, 같은 `thread_id`로 `Command(resume=...)`를 호출해 이어서 실행

핵심 파일:

- `app/main.py`: CLI 실행/재개 처리
- `app/graph.py`: 노드/엣지 조립, checkpointer 설정
- `app/retrieval.py`: 검색형 워크플로우
- `app/action.py`: 승인 기반 액션 워크플로우

## 사용 예시

### 1) Retrieval 요청

입력: `환불 정책 알려줘`

- Route가 `retrieval`로 분기
- 문서 검색/평가 후 답변 생성

### 2) Action 요청 (승인 필요)

입력: `고객에게 환불 처리 메일 보내줘`

- Route가 `action`으로 분기
- `approval_interrupt`에서 중단됨
- 승인/거절 입력 후 같은 `thread_id`로 재개

## 프로젝트 구조

```text
app/
  __init__.py
  main.py                 # CLI 데모 엔트리
  graph.py                # LangGraph 조립
  state.py                # State 정의
  routing.py              # 요청 분기/플래닝 로직
  retrieval.py            # Retrieval 노드
  action.py               # Action 노드 + interrupt
  tools.py                # 외부 도구 시뮬레이션
  vector_store.py         # 데모 벡터DB
```

## 확장 포인트

- 실제 LLM 연결: `routing.py` 의 규칙 기반 함수를 model 호출로 교체
- 실제 벡터DB 연결: `vector_store.py` 를 Chroma/Pinecone/pgvector로 교체
- 실제 도구 실행: `tools.py`를 메일/DB/API 클라이언트로 교체
- 모니터링: LangSmith 또는 OpenTelemetry 연동

## 벡터 DB 연결 방법 (실전 전환 가이드)

현재 코드는 `app/retrieval.py`에서 아래처럼 전역 저장소를 사용합니다.

- `vector_store = InMemoryVectorStore.bootstrap()`
- `retrieve_docs()`에서 `vector_store.search(...)` 호출

즉, **연결 지점은 `app/retrieval.py`의 `vector_store` 객체**입니다.  
실제 벡터 DB로 바꿀 때는 아래 2단계로 진행하면 됩니다.

1. `app/vector_store.py`에 실제 DB 클라이언트 래퍼 클래스 구현
   - 예: `class ChromaVectorStore`, `class PgVectorStore`
   - `search(query, top_k)` 메서드 시그니처를 동일하게 유지
2. `app/retrieval.py`에서 생성 객체만 교체
   - 예: `vector_store = ChromaVectorStore.from_env()`

권장 환경변수 예시:

- `VECTOR_DB_PROVIDER=chroma|pinecone|pgvector`
- `VECTOR_DB_URL=...`
- `VECTOR_DB_API_KEY=...` (필요 시)
- `VECTOR_DB_COLLECTION=...`

환경변수 설정 방법:

1. 터미널에서 임시 설정(현재 셸 세션에만 유효):

   ```bash
   export VECTOR_DB_PROVIDER=chroma
   export VECTOR_DB_URL=http://localhost:8000
   export VECTOR_DB_API_KEY=
   export VECTOR_DB_COLLECTION=my_docs
   ```

2. `.env` 파일로 설정(권장):

   ```env
   VECTOR_DB_PROVIDER=chroma
   VECTOR_DB_URL=http://localhost:8000
   VECTOR_DB_API_KEY=
   VECTOR_DB_COLLECTION=my_docs
   ```

   앱이 자동으로 `.env`를 읽지 않는 경우, 실행 전에 아래로 로드:

   ```bash
   set -a
   source .env
   set +a
   ```

3. Provider별 값 예시:
   - Chroma
     - `VECTOR_DB_PROVIDER=chroma`
     - `VECTOR_DB_URL=http://localhost:8000`
     - `VECTOR_DB_API_KEY=` (보통 비워둠)
     - `VECTOR_DB_COLLECTION=my_docs`
   - Pinecone
     - `VECTOR_DB_PROVIDER=pinecone`
     - `VECTOR_DB_URL=<pinecone 인덱스 endpoint>`
     - `VECTOR_DB_API_KEY=<pinecone api key>`
     - `VECTOR_DB_COLLECTION=<index 또는 namespace 이름>`
   - pgvector
     - `VECTOR_DB_PROVIDER=pgvector`
     - `VECTOR_DB_URL=postgresql://USER:PASS@HOST:5432/DBNAME`
     - `VECTOR_DB_API_KEY=` (보통 비워둠)
     - `VECTOR_DB_COLLECTION=<테이블/컬렉션 이름>`

운영 팁:

- 검색 결과 품질을 위해 문서 임베딩 생성 파이프라인(ingestion)을 별도 운영
- top_k/필터 조건/메타데이터 스키마(문서 id, 출처, 업데이트 시각)를 함께 설계
- 현재 `MemorySaver`는 인메모리이므로, 운영 환경에서는 영속 체크포인터 사용 권장
