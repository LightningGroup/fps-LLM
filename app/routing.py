from __future__ import annotations

from app.state import GraphState


def ingest(state: GraphState) -> GraphState:
    text = state["user_input"].strip()
    return {
        "normalized_input": " ".join(text.split()),
        "rewrite_count": 0,
    }


def route_request(state: GraphState) -> GraphState:
    text = state["normalized_input"].lower()

    if any(k in text for k in ["처리해", "실행", "보내", "생성", "등록"]):
        return {"route": "action", "route_reason": "외부 시스템 변경 가능성 있음"}
    if any(k in text for k in ["정책", "알려", "무엇", "설명", "찾아"]):
        return {"route": "retrieval", "route_reason": "지식 검색 기반 질의"}
    return {"route": "direct", "route_reason": "간단 질의로 직접 응답 가능"}


def direct_answer(state: GraphState) -> GraphState:
    return {
        "final_answer": (
            "직접 응답 경로로 처리했습니다. "
            f"질문: {state['normalized_input']}"
        )
    }


def route_after_ingest(state: GraphState) -> str:
    return state["route"]
