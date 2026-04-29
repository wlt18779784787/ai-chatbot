"""
CLI 聊天入口。
"""

from core.chat_core import SessionScope, chat_core


def run_cli():
    user_id = input("请输入 user_id: ").strip() or "user_001"
    agent_id = input("请输入 agent_id: ").strip() or "agent_cli"
    scope = SessionScope(user_id=user_id, agent_id=agent_id)

    print("=" * 50)
    print(
        "AI Chatbot (Mem0 + OpenRouter)"
        f" | user_id={scope.user_id} | agent_id={scope.agent_id}"
    )
    print("输入 quit 退出")
    print("=" * 50)

    while True:
        user_input = input("\n你：").strip()
        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "q"]:
            print("再见")
            break

        result = chat_core.chat(user_input, scope.user_id, scope.agent_id)
        if result["success"]:
            print(f"\nAI ({result['duration']:.2f}s)：{result['response']}")
            memories = chat_core.search_memories(user_input, scope)
            if memories:
                print(f"[记忆] 找到 {len(memories)} 条相关长期记忆")
        else:
            print(f"错误：{result['error']}")


if __name__ == "__main__":
    try:
        run_cli()
    finally:
        chat_core.shutdown()
        print("[系统] 线程池已关闭")
