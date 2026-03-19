from __future__ import annotations

import uuid
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langgraph.types import Command

from app.graph import build_app


app = FastAPI(title="LangGraph Orchestrator API", version="0.1.0")
_graph_app = build_app()


class StartThreadResponse(BaseModel):
    thread_id: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="사용자 질문/요청")
    thread_id: str | None = Field(
        default=None,
        description="세션 식별자. 없으면 서버가 새로 생성합니다.",
    )


class ChatResponse(BaseModel):
    thread_id: str
    status: Literal["completed", "approval_required"]
    final_answer: str | None = None
    interrupt: dict[str, Any] | None = None


class ApprovalRequest(BaseModel):
    thread_id: str
    decision: Literal["approved", "rejected"]


class ApprovalResponse(BaseModel):
    thread_id: str
    status: Literal["completed"]
    final_answer: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/threads", response_model=StartThreadResponse)
def start_thread() -> StartThreadResponse:
    return StartThreadResponse(thread_id=str(uuid.uuid4()))


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = _graph_app.invoke(
        {"user_input": req.message, "thread_id": thread_id},
        config=config,
    )

    if "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"]
        if isinstance(interrupt_payload, tuple) and len(interrupt_payload) > 0:
            value = getattr(interrupt_payload[0], "value", interrupt_payload)
        else:
            value = interrupt_payload

        return ChatResponse(
            thread_id=thread_id,
            status="approval_required",
            interrupt={"detail": value},
        )

    return ChatResponse(
        thread_id=thread_id,
        status="completed",
        final_answer=result.get("final_answer", str(result)),
    )


@app.post("/chat/approval", response_model=ApprovalResponse)
def approve_chat(req: ApprovalRequest) -> ApprovalResponse:
    config = {"configurable": {"thread_id": req.thread_id}}

    try:
        resumed = _graph_app.invoke(Command(resume=req.decision), config=config)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="유효한 인터럽트 상태를 찾지 못했습니다. thread_id를 확인하세요.",
        ) from exc

    return ApprovalResponse(
        thread_id=req.thread_id,
        status="completed",
        final_answer=resumed.get("final_answer", str(resumed)),
    )
