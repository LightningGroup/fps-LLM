from __future__ import annotations

from typing import Any, Literal, TypedDict


RouteType = Literal["direct", "retrieval", "action"]


class GraphState(TypedDict, total=False):
    # input/context
    user_input: str
    normalized_input: str
    thread_id: str

    # routing
    route: RouteType
    route_reason: str

    # retrieval workflow
    retrieval_query: str
    retrieved_docs: list[dict[str, Any]]
    doc_grade: Literal["sufficient", "insufficient"]
    rewrite_count: int
    answer_draft: str
    answer_checked: bool

    # action workflow
    action_plan: dict[str, Any]
    approval: Literal["approved", "rejected"]
    tool_result: dict[str, Any]
    observation: str

    # common outputs
    final_answer: str
