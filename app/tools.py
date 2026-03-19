from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def execute_external_tool(action_plan: dict[str, Any]) -> dict[str, Any]:
    """외부 도구/API 실행 시뮬레이터."""
    action = action_plan.get("action", "unknown")

    if action == "send_refund_email":
        return {
            "status": "success",
            "action": action,
            "message_id": "mail-20260319-0001",
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "detail": "환불 안내 메일 발송 완료",
        }

    return {
        "status": "noop",
        "action": action,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "detail": "정의되지 않은 액션이어서 실행 생략",
    }
