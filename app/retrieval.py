from __future__ import annotations

from app.state import GraphState
from app.vector_store import InMemoryVectorStore


vector_store = InMemoryVectorStore.bootstrap()


def plan_retrieval(state: GraphState) -> GraphState:
    return {"retrieval_query": state["normalized_input"]}


def retrieve_docs(state: GraphState) -> GraphState:
    docs = vector_store.search(state["retrieval_query"], top_k=3)
    return {"retrieved_docs": docs}


def grade_docs(state: GraphState) -> GraphState:
    docs = state.get("retrieved_docs", [])
    if len(docs) >= 1:
        return {"doc_grade": "sufficient"}
    return {"doc_grade": "insufficient"}


def rewrite_query(state: GraphState) -> GraphState:
    cnt = state.get("rewrite_count", 0) + 1
    rewritten = f"{state['retrieval_query']} 환불 정책 안내"
    return {
        "rewrite_count": cnt,
        "retrieval_query": rewritten,
    }


def generate_answer(state: GraphState) -> GraphState:
    docs = state.get("retrieved_docs", [])
    if not docs:
        draft = "관련 문서를 찾지 못했습니다. 추가 정보를 주시면 더 정확히 도와드릴게요."
    else:
        bullets = "\n".join(
            f"- ({d['id']}) {d['title']}: {d['text']}" for d in docs
        )
        draft = f"검색 결과 기반 답변입니다.\n{bullets}"

    return {"answer_draft": draft}


def answer_check(state: GraphState) -> GraphState:
    ok = "- (" in state.get("answer_draft", "")
    if ok:
        return {
            "answer_checked": True,
            "final_answer": state["answer_draft"] + "\n\n(근거 문서 포함)"
        }

    return {
        "answer_checked": False,
        "final_answer": state["answer_draft"] + "\n\n(근거 부족: 재질문 권장)"
    }


def route_retrieval_grade(state: GraphState) -> str:
    if state["doc_grade"] == "sufficient":
        return "generate_answer"

    # 최대 2회 재작성 후 종료
    if state.get("rewrite_count", 0) >= 2:
        return "generate_answer"
    return "rewrite_query"
