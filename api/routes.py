"""
FastAPI 路由层。
"""

import asyncio

from fastapi import APIRouter, Form, HTTPException, Query

from core.chat_core import SessionScope, chat_core

router = APIRouter()


@router.post("/api/chat")
async def chat(
    message: str = Form(...),
    user_id: str = Form(...),
    agent_id: str = Form(...),
):
    """聊天接口，要求 user_id 和 agent_id 都显式传入。"""
    if not user_id:
        raise HTTPException(status_code=400, detail="请先输入用户ID")
    if not agent_id:
        raise HTTPException(status_code=400, detail="请先输入Agent ID")

    result = await asyncio.to_thread(chat_core.chat, message, user_id, agent_id)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])

    return result


@router.get("/api/history/{user_id}")
async def get_history(user_id: str, agent_id: str = Query(...)):
    """按 user_id + agent_id 读取对应会话的短期历史。"""
    scope = SessionScope(user_id=user_id, agent_id=agent_id)
    session = chat_core.get_or_create_session(scope)
    with chat_core.sessions_lock:
        history = list(session.conversation_window)
    return {"history": history}
