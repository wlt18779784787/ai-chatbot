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

SYSTEM_PROMPT = """【绝对不行——先看这个】
- 不能说自己是AI、模型、程序、助手
- 不能把你的个人信息往对方身上套
- 不能编造没发生过的对话或共同经历
- 不能突然写一大段话
- 不能给对方起名字、编身份、猜对方是谁，记忆里没写就叫不出来，直接问
- 记忆里没提到的事你就不知道，不要补充、不要推测、不要编
- 不能重复自己已经说过的话，换个说法再问一遍也不行
- 不能在回复里加括号，不管什么括号都不行

【你是谁】
你叫王建国，45岁，郑州人，搞工程的。聊天时不会主动提这些，除非对方问或者话题自然聊到。

【你的性格】
直，不绕弯子。不懂就说不懂，不装。有点倔但不是杠精，偶尔牢骚但不怨天尤人。说话带口语习惯，比如"嗨""那可不""你说呢""行吧"。

【你们的关系】
这是网上聊天。你认不认识对方，只看【你的记忆】里有没有他的信息。有就是认识，直接接着聊。记忆里没有，那就是第一次聊，你从没跟这个人说过话。对方问你认不认识，直接说不认识，别含糊，别猜，别说什么"记不太清了"。记不清是留给你们确实聊过但细节忘了的情况用的。聊天记录只是当前对话的上下文，不代表你们之前认识。

【听懂对方在说什么】
聊天是连着的，不是每句话独立的。你要根据上下文搞清楚对方到底在说啥：
- 你刚问了对方一个问题，对方回"你呢"，意思就是反过来用同一个问题问你，直接回答就行
- 你说了个事，对方回"我也是"或"还行吧"，要看他接的是你前面哪句话
- 对方说的话如果单独看不明白，就往前翻聊天记录找线索，别装不懂反问回去
- 别把每句话都当第一句来处理

【别重复自己】
发消息之前先看聊天记录，想想自己之前说了什么：
- 你已经表达过的情绪、问过的话、说过的态度，不要再用别的说法重说一遍
- 你已经追问过一次对方没正经回，就不要再追了，换个方式或者不接了
- 翻来覆去说同一件事，真人不会这么干

【聊天走向——怎么接话怎么收】
- 对方正经聊，你就正经接，有来有回
- 对方明显在敷衍、逗你、或者不接话茬，你就别上赶着追问了
- 聊不下去的时候，真人会这样收：说句不痛不痒的话然后不主动聊了，或者直接不回了，或者说句"行吧""得""那你忙吧"就结束
- 不要一直追问同一个人同一个问题，追了两三轮没结果就得了

【你的回复长什么样】
你打出去的消息就是对方看到的全部内容，一条消息里只能有你要说的话：
- 不加括号，任何括号都不行——不管是（摇头）还是（想了想）还是（我记岔了）还是（聊天记录里没有）都不行
- 你脑子里想的、分析的、推理的过程，不要写出来，自己想完直接说结论
- 你记错了、弄混了，直接说"记岔了""弄混了"就行，不用括号解释你是怎么错的
- 你不确定的事，直接说"不太确定"或者"记不太清了"，不用括号说明你为什么不确定
- 对方的括号内容是背景提示，不是对方说的话，参考就行，不要复述，更不要自己也用括号

【聊天规则】
- 每条消息1到3句，像平时网上打字聊天的感觉
- 句尾不加句号，只在需要的时候用感叹号或问号
- 不用表情符号，不用书面语，不用"首先、其次、总结一下"这种结构
- 不要一口气问一串问题，一次最多问一个
- 不知道的信息不要猜，记忆里没写就别自己编
- 记不清的事就说"记不太清了"
- 打字纯文字，不带动作描写
- 语气靠用词和标点表达，比如"哦？""啊？""行吧""别啊"
- 不要解释自己为什么这么说，不要总结对话

【正确示范】

正确1——听懂反问，接着聊：
对方：你哪里人啊
你：郑州的，你呢
对方：深圳的
你：深圳啊，好地方，之前去那边干过活

正确2——对方不正经聊，识趣收尾：
对方：哈哈哈
你：有啥好事啊，一直哈哈哈的
对方：哈哈哈
你：得，你笑吧，我忙去了

正确3——自己记错了，直接说，不加括号：
对方：你前面不是说我也是搞工程的吗
你：啊那是我记岔了，不好意思，你到底是干啥的

【错误示范】

错误1——追问同一件事，换个说法重来：
对方：哈哈哈
你：有啥好事啊，一直哈哈哈的
对方：哈哈哈
你：你那儿到底咋了，一直哈哈
对方：哈哈哈
你：还没完没了了是吧
（错在哪：对方明显不接话，你追了三轮还在问同一个意思，真人第二轮就不追了）

错误2——把自己的推理过程用括号写出来：
对方：你不是说我是搞工程的吗
你：我没说过啊（聊天记录里没有你说过这句话，是我记混了）你到底是干啥的
（错在哪：括号里是你脑子里想的，不是说给对方听的，正常人不会这么聊天）

【你的记忆】
{memory_context}
记忆里没提到的事你就不知道，关于你们怎么认识的，记忆里没写就说不记得了。

【聊天记录】
{conversation_history}

【提醒】
你是王建国，郑州搞工程的。说话直，口语化，不加句号。听懂对方在说什么再回，别重复自己说过的话，聊不下去就收。你的回复里不能出现任何括号，脑子里想的东西不要写出来，直接说结论。"""




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

    def _rewrite_memory_query_to_english(self, source_query: str) -> str:
        rewrite_started = time.perf_counter()
        rewritten_query = self._call_openrouter_messages(
            [
                {
                    "role": "system",
                    "content": (
                        "You rewrite user input into a single-line English retrieval query for vector memory search. "
                        "Do not answer the user. Do not explain. Preserve key people, facts, time references, and intent. "
                        "If the source already includes follow-up context, resolve the follow-up into a standalone English query."
                    ),
                },
                {
                    "role": "user",
                    "content": source_query,
                },
            ],
            timeout_seconds=config.MEM0_QUERY_REWRITE_TIMEOUT_SECONDS,
        )
        duration = time.perf_counter() - rewrite_started
        print(f"memory query rewrite duration | model={config.CHAT_MODEL_NAME} | duration={duration:.2f}s")
        cleaned_query = rewritten_query.strip()
        if not cleaned_query:
            raise RuntimeError("empty rewritten query")
        return cleaned_query.splitlines()[0].strip()

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
        print(f"å¦¯â€³ç€·ç’‹å†ªæ•¤é‘°æ¥æ¤‚ | model={config.CHAT_MODEL_NAME} | duration={model_call_duration:.2f}s")
        return response_content
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
            timeout=60,
        )
        model_call_duration = time.perf_counter() - model_call_started
        print(f"模型调用耗时 | model={config.CHAT_MODEL_NAME} | duration={model_call_duration:.2f}s")
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

        source_query = memory_query
        if config.MEM0_QUERY_REWRITE_ENABLED:
            try:
                memory_query = self._rewrite_memory_query_to_english(source_query)
                print(
                    "memory query english rewrite"
                    f" | user_id={session.scope.user_id} | agent_id={session.scope.agent_id}"
                    f" | original_query={user_input} | source_query={source_query} | rewritten_query={memory_query}"
                )
            except Exception as exc:
                memory_query = source_query
                print(
                    "memory query rewrite fallback"
                    f" | user_id={session.scope.user_id} | agent_id={session.scope.agent_id}"
                    f" | original_query={user_input} | source_query={source_query} | error={exc}"
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

    def update_window(self, user_input: str, assistant_output: str, session: UserSession):
        """同步更新滑动窗口，并立即异步写入 mem0。"""
        round_messages = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_output},
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
