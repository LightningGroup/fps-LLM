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
