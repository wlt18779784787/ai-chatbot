"""
核心聊天逻辑。
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests
from mem0 import Memory

from config import config, get_mem0_oss_config, get_openrouter_reasoning_config
from core.mem0_compat import apply_mem0_milvus_dense_only_patch, apply_mem0_openrouter_reasoning_config_patch

# 用线程池承接较慢的 IO 操作，避免把记忆保存阻塞在主请求路径里。
executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)

# 启动时先打补丁，关闭 mem0 在 Milvus 上的 BM25 / sparse 索引逻辑。
apply_mem0_milvus_dense_only_patch()
apply_mem0_openrouter_reasoning_config_patch()

INITIAL_MEMORY_BATCH_ROUNDS = 5
MEMORY_BATCH_ROUNDS = 5
MEMORY_BATCH_OVERLAP_ROUNDS = 2
FOLLOW_UP_EXACT_PHRASES = {
    "还有呢",
    "还有吗",
    "还有什么",
    "然后呢",
    "继续",
    "接着说",
    "再说说",
    "展开说说",
    "别的呢",
    "后面呢",
    "然后",
}

SYSTEM_PROMPT = """【你是谁】你叫王建国，45岁，郑州人，搞工程的。聊天时不会主动提这些，除非对方问或者话题自然聊到。
【你的性格】直，不绕弯子。不懂就说不懂，不装。有点倔但不是杠精，偶尔牢骚但不怨天尤人。说话带口语习惯，比如“嗨”“那可不”“你说呢”“行吧”。
【你们的关系】这是网上聊天。你认不认识对方，只看【你的记忆】里有没有他的信息。有就是认识，直接接着聊。记忆里没有，那就是第一次聊，你从没跟这个人说过话。对方问你认不认识，直接说不认识，别含糊，别猜，别说什么“记不太清了”。记不清是留给你们确实聊过但细节忘了的情况用的。聊天记录只是当前对话的上下文，不代表你们之前认识。
【聊天规则】
- 每条消息1到3句，像平时网上打字聊天的感觉
- 句尾不加句号，只在需要的时候用感叹号或问号
- 不用表情符号，不用书面语，不用“首先、其次、总结一下”这种结构
- 不要一口气问一串问题，一次最多问一个
- 不知道的信息不要猜，记忆里没写就别自己编
- 记不清的事就说“记不太清了”
- 打字纯文字，不带动作描写，不要出现（摇头）（叹气）（想了想）这种东西
- 语气靠用词和标点表达，比如“哦？”“啊？”“行吧”“别啊”
- 聊天记录里如果有括号括起来的内容，那是背景信息不是对方说的话，你可以参考来接话，但不要复述，不要用括号回复
- 不要解释自己为什么这么说，不要总结对话

【你的记忆】
{memory_context}
记忆里没提到的事你就不知道，不要补充、不要推测、不要编。关于你们怎么认识的，记忆里没写就说不记得了。
【聊天记录】
{conversation_history}

【绝对不行】
- 不能说自己是AI、模型、程序、助手
- 不能把你的个人信息往对方身上套
- 不能编造没发生过的对话或共同经历
- 不能突然写一大段话
- 不能给对方起名字、编身份、猜对方是谁，记忆里没写就叫不出来，直接问对方是谁"""


@dataclass(frozen=True)
class SessionScope:
    """会话作用域，按 user_id + agent_id 决定检索和存储范围。"""

    user_id: str
    agent_id: str


@dataclass
class UserSession:
    """用户会话，保存短期窗口和待写入 mem0 的轮次缓冲。"""

    scope: SessionScope
    conversation_window: List[dict] = field(default_factory=list)
    memory_round_buffer: List[List[dict]] = field(default_factory=list)
    memory_flush_count: int = 0


class ChatCore:
    """聊天主流程。"""

    def __init__(self):
        self.sessions: Dict[SessionScope, UserSession] = {}
        self.sessions_lock = threading.Lock()
        self.executor = executor
        self.memory_client: Optional[Memory] = Memory.from_config(get_mem0_oss_config())

    def get_or_create_session(self, scope: SessionScope) -> UserSession:
        """按作用域取会话，没有就新建一个。"""
        with self.sessions_lock:
            if scope not in self.sessions:
                self.sessions[scope] = UserSession(scope=scope)
            return self.sessions[scope]

    def _is_follow_up_query(self, user_input: str) -> bool:
        """判断当前输入是否属于依赖上一轮语境的续问。"""
        normalized = user_input.strip().lower()
        if not normalized:
            return False
        if normalized in FOLLOW_UP_EXACT_PHRASES:
            return True
        if len(normalized) <= 6 and any(token in normalized for token in ["还有", "继续", "然后", "别的", "后面"]):
            return True
        return False

    def _get_last_completed_round(self, session: UserSession) -> Optional[Tuple[str, str]]:
        """从窗口里取最近一轮完整 user+assistant 对话。"""
        with self.sessions_lock:
            window_snapshot = list(session.conversation_window)

        if len(window_snapshot) < 2:
            return None

        last_user = None
        last_assistant = None
        for message in reversed(window_snapshot):
            if last_assistant is None and message.get("role") == "assistant":
                last_assistant = message.get("content", "")
                continue
            if last_assistant is not None and message.get("role") == "user":
                last_user = message.get("content", "")
                break

        if last_user and last_assistant:
            return last_user, last_assistant
        return None

    def _build_memory_query(self, user_input: str, session: UserSession) -> Tuple[str, bool]:
        """构造用于 mem0 检索的 query。"""
        if not self._is_follow_up_query(user_input):
            return user_input, False

        last_round = self._get_last_completed_round(session)
        if not last_round:
            return user_input, False

        last_user, last_assistant = last_round
        rewritten_query = (
            "请围绕上一轮话题继续检索相关长期记忆。"
            f" 上一轮用户问题：{last_user}"
            f" 上一轮助手回复：{last_assistant}"
            f" 当前用户追问：{user_input}"
        )
        return rewritten_query, True

    def search_memories(self, query: str, scope: SessionScope, top_k: int = 7) -> List[dict]:
        """从 mem0 检索相关长期记忆，使用 user_id + agent_id 过滤。"""
        try:
            filters = {
                "user_id": scope.user_id,
                "agent_id": scope.agent_id,
            }
            result = self.memory_client.search(query, filters=filters, top_k=top_k)
            return result.get("results", [])
        except Exception as exc:
            print(f"记忆检索失败，已降级为空记忆 | user_id={scope.user_id} | agent_id={scope.agent_id} | error={exc}")
            return []

    def save_to_memory(self, messages: List[dict], scope: SessionScope):
        """把多轮消息批量写入 mem0，并附带 user_id + agent_id 作用域。"""

        self.memory_client.add(
            messages,
            user_id=scope.user_id,
            agent_id=scope.agent_id,
        )

    def _persist_memory_batch(self, messages: List[dict], scope: SessionScope, rounds_count: int):
        """在线程池里异步执行的批量记忆写入任务。"""
        try:
            self.save_to_memory(messages, scope)
            print(
                "记忆批量保存已提交"
                f" | user_id={scope.user_id} | agent_id={scope.agent_id}"
                f" | rounds={rounds_count} | messages={len(messages)}"
            )
        except Exception as exc:
            print(f"记忆保存失败，已忽略 | user_id={scope.user_id} | agent_id={scope.agent_id} | error={exc}")

    def call_model(self, messages: List[dict]) -> str:
        """调用 OpenRouter 聊天模型。"""
        model_call_started = time.perf_counter()
        request_payload = {
            "model": config.MODEL_NAME,
            "messages": messages,
        }
        reasoning = get_openrouter_reasoning_config()
        if reasoning is not None:
            request_payload["reasoning"] = reasoning

        response = requests.post(
            config.OPENROUTER_CHAT_URL,
            headers={
                "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json=request_payload,
            timeout=60,
        )
        model_call_duration = time.perf_counter() - model_call_started
        print(f"模型调用耗时 | model={config.MODEL_NAME} | duration={model_call_duration:.2f}s")
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _get_current_time_text(self) -> str:
        """生成当前时间文本，固定使用中国时区，避免模型误判记忆时间。"""
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        return now.strftime("%Y-%m-%d %H:%M:%S CST")

    def _build_messages_with_timings(self, user_input: str, session: UserSession) -> Tuple[List[dict], Dict[str, float]]:
        """拼装 system prompt 和用户输入，并返回构建阶段耗时。"""
        timings: Dict[str, float] = {}

        memory_query, was_rewritten = self._build_memory_query(user_input, session)
        if was_rewritten:
            print(
                "记忆检索续问改写"
                f" | user_id={session.scope.user_id} | agent_id={session.scope.agent_id}"
                f" | original_query={user_input} | rewritten_query={memory_query}"
            )

        memory_search_started = time.perf_counter()
        memories = self.search_memories(memory_query, session.scope)
        memory_search_duration = time.perf_counter() - memory_search_started
        timings["memory_search_duration"] = memory_search_duration
        print(f"记忆检索耗时 | user_id={session.scope.user_id} | agent_id={session.scope.agent_id} | duration={memory_search_duration:.2f}s")

        prompt_build_started = time.perf_counter()
        memory_context = "\n".join([f"- {item['memory']}" for item in memories]) if memories else "暂无"
        current_time_text = self._get_current_time_text()

        with self.sessions_lock:
            window_snapshot = list(session.conversation_window)

        if window_snapshot:
            conversation_history = "\n".join(
                [f"{'我' if item['role'] == 'user' else '王建国'}：{item['content']}" for item in window_snapshot]
            )
        else:
            conversation_history = "暂无"

        system_prompt = (
            f"【当前时间】{current_time_text}\n"
            "【时间要求】回答时要考虑当前时间，不要把过期记忆当成当前事实；"
            "如果记忆里涉及时间但不确定是否仍然有效，要明确说可能是之前的情况。\n\n"
            + SYSTEM_PROMPT.format(
                conversation_history=conversation_history,
                memory_context=memory_context,
            )
        )

        timings["prompt_build_duration"] = time.perf_counter() - prompt_build_started

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ], timings

    def build_messages(self, user_input: str, session: UserSession) -> List[dict]:
        """兼容旧调用，只返回 messages。"""
        messages, _ = self._build_messages_with_timings(user_input, session)
        return messages

    def _print_prompt_messages(self, scope: SessionScope, messages: List[dict]):
        """把本轮实际发送给模型的提示词完整打印到控制台。"""
        print("\n" + "=" * 80)
        print(f"本轮模型请求提示词 | user_id={scope.user_id} | agent_id={scope.agent_id}")
        for index, message in enumerate(messages, start=1):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            print("-" * 80)
            print(f"[{index}] role={role}")
            print(content)
        print("=" * 80 + "\n")

    def _collect_memory_batch_if_ready(self, session: UserSession) -> Optional[Tuple[List[dict], int]]:
        """按每批 5 轮、保留末尾 2 轮重叠的规则收集待写入批次。"""
        buffer = session.memory_round_buffer
        threshold = INITIAL_MEMORY_BATCH_ROUNDS if session.memory_flush_count == 0 else MEMORY_BATCH_ROUNDS
        if len(buffer) < threshold:
            return None

        rounds_to_persist = buffer[:threshold]
        flattened_messages = [message for round_messages in rounds_to_persist for message in round_messages]
        session.memory_flush_count += 1
        session.memory_round_buffer = list(buffer[-MEMORY_BATCH_OVERLAP_ROUNDS:])
        return flattened_messages, threshold

    def update_window(self, user_input: str, assistant_output: str, session: UserSession):
        """同步更新滑动窗口，并在满足批量规则时异步写入 mem0。"""
        round_messages = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_output},
        ]

        batch_to_persist: Optional[Tuple[List[dict], int]] = None
        with self.sessions_lock:
            session.conversation_window.extend(round_messages)
            while len(session.conversation_window) > config.WINDOW_SIZE:
                session.conversation_window.pop(0)
                session.conversation_window.pop(0)

            session.memory_round_buffer.append(round_messages)
            batch_to_persist = self._collect_memory_batch_if_ready(session)
            print(
                "记忆缓冲状态"
                f" | user_id={session.scope.user_id} | agent_id={session.scope.agent_id}"
                f" | buffered_rounds={len(session.memory_round_buffer)} | flushed_batches={session.memory_flush_count}"
            )

        if batch_to_persist is not None:
            batch_messages, rounds_count = batch_to_persist
            self.executor.submit(self._persist_memory_batch, batch_messages, session.scope, rounds_count)

    def chat(self, user_input: str, user_id: str, agent_id: str) -> dict:
        """处理单轮聊天请求。"""
        scope = SessionScope(user_id=user_id, agent_id=agent_id)
        request_started = time.perf_counter()

        try:
            session = self.get_or_create_session(scope)
            messages, _ = self._build_messages_with_timings(user_input, session)
            self._print_prompt_messages(scope, messages)

            response = self.call_model(messages)

            self.update_window(user_input, response, session)
            duration = time.perf_counter() - request_started

            return {
                "success": True,
                "response": response,
                "duration": round(duration, 2),
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def shutdown(self):
        """关闭线程池，确保后台任务收尾。"""
        self.executor.shutdown(wait=True)


chat_core = ChatCore()
