from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.action import (
    approval_interrupt,
    execute_tool,
    finalize_answer,
    observe_result,
    plan_action,
    route_after_approval,
)
from app.retrieval import (
    answer_check,
    generate_answer,
    grade_docs,
    plan_retrieval,
    retrieve_docs,
    rewrite_query,
    route_retrieval_grade,
)
from app.routing import direct_answer, ingest, route_after_ingest, route_request
from app.state import GraphState


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    # common
    graph.add_node("ingest", ingest)
    graph.add_node("route", route_request)
    graph.add_node("direct", direct_answer)

    # retrieval workflow
    graph.add_node("plan_retrieval", plan_retrieval)
    graph.add_node("retrieve_docs", retrieve_docs)
    graph.add_node("grade_docs", grade_docs)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("answer_check", answer_check)

    # action workflow
    graph.add_node("plan_action", plan_action)
    graph.add_node("approval_interrupt", approval_interrupt)
    graph.add_node("execute_tool", execute_tool)
    graph.add_node("observe_result", observe_result)
    graph.add_node("finalize_answer", finalize_answer)

    # edges
    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "route")

    graph.add_conditional_edges(
        "route",
        route_after_ingest,
        {
            "direct": "direct",
            "retrieval": "plan_retrieval",
            "action": "plan_action",
        },
    )

    graph.add_edge("direct", END)

    graph.add_edge("plan_retrieval", "retrieve_docs")
    graph.add_edge("retrieve_docs", "grade_docs")
    graph.add_conditional_edges(
        "grade_docs",
        route_retrieval_grade,
        {
            "rewrite_query": "rewrite_query",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("rewrite_query", "retrieve_docs")
    graph.add_edge("generate_answer", "answer_check")
    graph.add_edge("answer_check", END)

    graph.add_edge("plan_action", "approval_interrupt")
    graph.add_conditional_edges(
        "approval_interrupt",
        route_after_approval,
        {
            "execute_tool": "execute_tool",
            "finalize_answer": "finalize_answer",
        },
    )
    graph.add_edge("execute_tool", "observe_result")
    graph.add_edge("observe_result", "finalize_answer")
    graph.add_edge("finalize_answer", END)

    return graph


def build_app():
    return build_graph().compile(checkpointer=MemorySaver())
