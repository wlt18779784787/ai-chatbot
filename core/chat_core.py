"""
核心聊天逻辑。
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
import re

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

QUERY_REWRITE_PROMPT = """
你是社交聊天AI的记忆检索 query 生成器。

你的任务不是回答用户，也不是解释上下文。
你的任务是把【当前用户问题】改写成一句适合查询长期记忆的中文检索 query。

核心规则：
- 只输出一句中文 query
- 不要输出分析过程
- 不要输出“说明、表示、需要、判断、上下文中、未知”等解释性文字
- 不要说用户身份未知
- 不要回答用户问题
- 不要编造记忆里没有的信息
- 如果用户问“我是谁”“你认识我吗”“你记得我吗”，统一改写为：查询当前用户的身份、昵称、关系和历史互动记忆
- 如果当前问题本身已经清楚，就保留核心意思
- 如果有明确上下文指代，才补全人物、事件、时间
- 不确定的内容不要补

最近两轮对话：
{conversation_history}

当前用户问题：
{user_input}

只输出最终检索 query：
""".strip()


SYSTEM_PROMPT = """
你叫王建国，45岁，郑州人，搞工程的。
除非自然聊到，否则别主动介绍自己。

说话直，口语化，不装，不绕弯子。
可用：嗨、行吧、得、那可不、你说呢。

规则：
- 不能说自己是AI、助手、模型、程序
- 不能编造记忆、经历、关系
- 记忆没有就不知道，别猜
- 不能把自己背景套对方身上
- 不重复自己说过的话
- 不长篇大论
- 不用任何括号

关系判断：
- 是否认识对方，只看记忆
- 有记忆：按认识的人聊天
- 没记忆：默认第一次聊
- 当前聊天记录不代表以前认识
- 对方问认不认识，没记忆就直接说不认识

聊天理解：
- 结合上下文，别把每句话当独立问题
- “你呢”通常是反问
- “我也是”要结合上一句理解
- 能懂就直接接，别装傻

聊天节奏：
- 对方正常聊，你正常接
- 对方敷衍，就别硬追问
- 同一问题追一次没回应就收
- 可收尾：行吧、得、那你忙吧

回复格式：
- 每次1~2句
- 简短，像真人聊天
- 不写小作文
- 不用总结式表达
- 不加句号
- 可用问号、感叹号
- 最多问一个问题
- 不用表情
- 不写动作
- 不暴露推理过程

你的记忆：
{memory_context}

当前聊天记录：
{conversation_history}

直接回复用户。
"""



@dataclass(frozen=True)
class SessionScope:
    """会话作用域，按 user_id + agent_id 决定检索和存储范围。"""

    user_id: str
    agent_id: str


@dataclass
class UserSession:
    """用户会话，保存短期滑动窗口。"""

    scope: SessionScope
    conversation_window: List[dict] = field(default_factory=list)


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

    def _get_window_snapshot(self, session: UserSession) -> List[dict]:
        """返回当前会话窗口快照，避免持锁拼装 prompt。"""
        with self.sessions_lock:
            return list(session.conversation_window)

    def _get_recent_complete_round_messages(self, session: UserSession, rounds: int = 2) -> List[dict]:
        """返回最近 N 组完整 user+assistant 问答，默认取最近两轮。"""
        if rounds <= 0:
            return []

        window_snapshot = self._get_window_snapshot(session)
        recent_messages: List[dict] = []
        round_count = 0
        index = len(window_snapshot) - 1

        while index > 0 and round_count < rounds:
            assistant_message = window_snapshot[index]
            user_message = window_snapshot[index - 1]

            if assistant_message.get("role") == "assistant" and user_message.get("role") == "user":
                recent_messages.insert(0, user_message)
                recent_messages.insert(1, assistant_message)
                round_count += 1
                index -= 2
                continue

            index -= 1

        return recent_messages

    def _format_conversation_history(self, window_snapshot: List[dict], user_label: str = "用户", assistant_label: str = "王建国") -> str:
        """把窗口快照格式化为 prompt 文本。"""
        if not window_snapshot:
            return "暂无"

        lines = []
        for item in window_snapshot:
            role = item.get("role")
            if role == "user":
                speaker = user_label
            elif role == "assistant":
                speaker = assistant_label
            else:
                speaker = role or "unknown"
            lines.append(f"{speaker}：{item.get('content', '')}")
        return "\n".join(lines)

    def _call_openrouter_messages(self, messages: List[dict], timeout_seconds: int) -> str:
        request_payload = {
            "model": config.CHAT_MODEL_NAME,
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
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _rewrite_memory_query(self, user_input: str, session: UserSession) -> str:
        rewrite_started = time.perf_counter()
        recent_round_messages = self._get_recent_complete_round_messages(session, rounds=2)
        prompt = QUERY_REWRITE_PROMPT.format(
            conversation_history=self._format_conversation_history(recent_round_messages),
            user_input=user_input,
        )
        rewritten_query = self._call_openrouter_messages(
            [
                {
                    "role": "system",
                    "content": prompt,
                },
            ],
            timeout_seconds=config.MEM0_QUERY_REWRITE_TIMEOUT_SECONDS,
        )
        duration = time.perf_counter() - rewrite_started
        cleaned_lines = [line.strip() for line in rewritten_query.splitlines() if line.strip()]
        if not cleaned_lines:
            raise RuntimeError("empty rewritten query")
        cleaned_query = cleaned_lines[0]
        print(
            "memory query rewrite"
            f" | user_id={session.scope.user_id} | agent_id={session.scope.agent_id}"
            f" | original_query={user_input} | duration={duration:.2f}s | rewritten_query={cleaned_query}"
        )
        return cleaned_query

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
        """把消息写入 mem0，附带 user_id + agent_id 作用域，由 AI 异步总结提炼。"""
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
        response_content = self._call_openrouter_messages(messages, timeout_seconds=60)
        model_call_duration = time.perf_counter() - model_call_started
        print(f"模型调用耗时 | model={config.CHAT_MODEL_NAME} | duration={model_call_duration:.2f}s")
        return response_content

    def _get_current_time_text(self) -> str:
        """生成当前时间文本，固定使用中国时区，避免模型误判记忆时间。"""
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        return now.strftime("%Y-%m-%d %H:%M:%S CST")

    def _build_messages_with_timings(self, user_input: str, session: UserSession) -> Tuple[List[dict], Dict[str, float]]:
        """拼装 system prompt 和用户输入，并返回构建阶段耗时。"""
        timings: Dict[str, float] = {}
        source_query = user_input
        window_snapshot = self._get_window_snapshot(session)
        if config.MEM0_QUERY_REWRITE_ENABLED:
            try:
                memory_query = self._rewrite_memory_query(user_input, session)
            except Exception as exc:
                memory_query = source_query
                print(
                    "memory query rewrite fallback"
                    f" | user_id={session.scope.user_id} | agent_id={session.scope.agent_id}"
                    f" | original_query={user_input} | window_size={len(window_snapshot)}"
                    f" | source_query={source_query} | error={exc}"
                )
        else:
            memory_query = source_query

        memory_search_started = time.perf_counter()
        memories = self.search_memories(memory_query, session.scope)
        memory_search_duration = time.perf_counter() - memory_search_started
        timings["memory_search_duration"] = memory_search_duration
        print(f"记忆检索耗时 | user_id={session.scope.user_id} | agent_id={session.scope.agent_id} | duration={memory_search_duration:.2f}s")

        prompt_build_started = time.perf_counter()
        memory_context = "\n".join([f"- {item['memory']}" for item in memories]) if memories else "暂无"
        current_time_text = self._get_current_time_text()
        conversation_history = self._format_conversation_history(
            window_snapshot,
            user_label="我",
            assistant_label="王建国",
        )

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

    def _normalize_assistant_window_text(self, text: str) -> str:
        """把助手写入滑动窗口的文本压成单行，避免单条回复内部断行。"""
        return re.sub(r"\s+", " ", text).strip()

    def update_window(self, user_input: str, assistant_output: str, session: UserSession):
        """同步更新滑动窗口，并立即异步写入 mem0。"""
        round_messages = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": self._normalize_assistant_window_text(assistant_output)},
        ]

        with self.sessions_lock:
            session.conversation_window.extend(round_messages)
            while len(session.conversation_window) > config.WINDOW_SIZE:
                session.conversation_window.pop(0)
                session.conversation_window.pop(0)

        self.executor.submit(self._persist_memory_batch, round_messages, session.scope, 1)

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
