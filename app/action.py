from __future__ import annotations

from langgraph.types import interrupt

from app.state import GraphState
from app.tools import execute_external_tool


def plan_action(state: GraphState) -> GraphState:
    text = state["normalized_input"]

    if "환불" in text and "메일" in text:
        plan = {
            "action": "send_refund_email",
            "target": "customer",
            "reason": "사용자 요청에 따른 환불 안내",
        }
    else:
        plan = {
            "action": "unknown",
            "target": "n/a",
            "reason": "명확한 실행 액션 미확정",
        }

    return {"action_plan": plan}


def approval_interrupt(state: GraphState) -> GraphState:
    payload = {
        "type": "approval_required",
        "message": "외부 실행 전 승인 필요",
        "action_plan": state["action_plan"],
        "guide": "approved 또는 rejected 중 하나를 입력하세요.",
    }

    decision = interrupt(payload)
    normalized = str(decision).strip().lower()
    if normalized not in {"approved", "rejected"}:
        normalized = "rejected"

    return {"approval": normalized}  # type: ignore[return-value]


def route_after_approval(state: GraphState) -> str:
    return "execute_tool" if state.get("approval") == "approved" else "finalize_answer"


def execute_tool(state: GraphState) -> GraphState:
    result = execute_external_tool(state["action_plan"])
    return {"tool_result": result}


def observe_result(state: GraphState) -> GraphState:
    r = state.get("tool_result", {})
    summary = f"action={r.get('action')} status={r.get('status')} detail={r.get('detail')}"
    return {"observation": summary}


def finalize_answer(state: GraphState) -> GraphState:
    if state.get("approval") == "rejected":
        return {
            "final_answer": "요청된 작업은 승인 거절로 실행하지 않았습니다. 필요하면 계획을 수정해 다시 요청해주세요."
        }

    if state.get("tool_result"):
        return {
            "final_answer": (
                "Action workflow 실행을 완료했습니다.\n"
                f"결과: {state['observation']}"
            )
        }

    return {"final_answer": "실행 가능한 액션을 찾지 못해 작업을 종료했습니다."}
