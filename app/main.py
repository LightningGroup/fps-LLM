from __future__ import annotations

import uuid

from langgraph.types import Command

from app.graph import build_app


def run_once(app, text: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    out = app.invoke({"user_input": text, "thread_id": thread_id}, config=config)
    return out


def main() -> None:
    app = build_app()
    thread_id = str(uuid.uuid4())

    print(f"thread_id={thread_id}")
    print("질문을 입력하세요. 예) 환불 정책 알려줘 / 고객에게 환불 처리 메일 보내줘")
    user_input = input("> ").strip()

    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"user_input": user_input, "thread_id": thread_id}, config=config)

    # interrupt 발생 시 __interrupt__ 키에 payload가 담겨 반환됨
    if "__interrupt__" in result:
        print("\n[승인 대기]")
        print(result["__interrupt__"])
        decision = input("승인하시겠습니까? (approved/rejected): ").strip().lower()
        resumed = app.invoke(Command(resume=decision), config=config)
        print("\n[최종 응답]")
        print(resumed.get("final_answer", resumed))
        return

    print("\n[최종 응답]")
    print(result.get("final_answer", result))


if __name__ == "__main__":
    main()
