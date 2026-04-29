# AI Chatbot

当前项目是一个基于 `FastAPI + Mem0 OSS + Milvus + OpenRouter` 的多用户聊天应用。

项目里有两层记忆：
- 短期记忆：进程内滑动窗口，保存在 `UserSession.conversation_window`
- 长期记忆：由 `mem0` 管理，写入 `Milvus`，并在本地 `SQLite` 中保存辅助数据

## Current Architecture

当前请求主链路如下：

1. 前端或 CLI 发起聊天请求
2. `chat_core` 按 `user_id + agent_id` 二元作用域获取进程内会话
3. 每一轮固定执行 `mem0.search(...)`
4. 检索时使用扁平二键 filters，要求 `user_id`、`agent_id` 同时匹配
5. 如果当前输入是“还有呢”“继续”“然后呢”这类弱语义续问，并且窗口里存在最近一轮完整问答，则先用最近一轮上下文改写检索 query
6. 组装带有当前中国时间的 `system prompt`
7. 调用 OpenRouter 聊天模型获取回复
8. 同步更新滑动窗口
9. 把当前轮加入记忆批量缓冲
10. 每累计满 5 轮后异步写入 `mem0`
11. 后续每新增 3 轮，按 5 轮窗口异步写入，其中前 2 轮与上一批重叠

## Memory Strategy

### 短期记忆

短期记忆保存在：
- `UserSession.conversation_window`

特点：
- 只存在内存里
- 作用域按 `user_id + agent_id` 隔离
- 服务重启后丢失
- 只用于当前对话上下文

### 长期记忆

长期记忆由 `mem0 OSS` 管理。

当前写入策略不是“每轮都调一次 `mem0.add(...)`”，而是：
- 每批：累计满 5 个用户-助手轮次后，批量写入这 5 轮
- 后续批次：每次写入 5 轮，其中前 2 轮是上一批末尾 2 轮
- 写入作用域同时带上：
  - `user_id`
  - `agent_id`

## Retrieval Behavior

当前长期记忆检索规则如下：

- 每一轮都会执行 `mem0.search(...)`
- 默认直接使用当前用户输入作为检索 query
- 如果当前输入是弱语义续问，例如：
  - `还有呢`
  - `继续`
  - `然后呢`
  - `别的呢`
  - `再说说`
- 并且最近一轮完整问答存在，则会把以下信息拼成新的检索 query：
  - 上一轮用户问题
  - 上一轮助手回复
  - 当前追问
- 这样可以改善“还有呢”这类输入直接拿去做向量检索时语义过空的问题

检索过滤条件固定为：

```python
filters = {
    "user_id": user_id,
    "agent_id": agent_id,
}
```

也就是只有 `user_id` 和 `agent_id` 同时匹配的记忆才会被查到。

## Milvus Usage

主 collection 名来自：
- `MILVUS_COLLECTION_NAME`

默认值：

```text
mem0_qwen4b
```

除了主 collection，`mem0` 还会自动派生一个实体索引 collection：

```text
mem0_qwen4b_entities
```

作用分工：
- 主 collection：存正式长期记忆
- `_entities` collection：存实体索引及其关联到哪些 `memory_id`

当前项目已在 `core/mem0_compat.py` 中给 `mem0` 的 Milvus 适配器打补丁，关闭了：
- `BM25`
- `sparse` 字段
- `SPARSE_INVERTED_INDEX`

所以现在实际使用的是纯 dense vector 检索。

## SQLite Usage

`mem0` 还会维护一个本地 SQLite 文件，路径来自：
- `MEM0_HISTORY_DB_PATH`

默认值：

```text
history.db
```

至少会有两张表：

### history

记录记忆对象的变更历史，常见字段包括：
- `memory_id`
- `old_memory`
- `new_memory`
- `event`
- `created_at`
- `updated_at`

### messages

缓存某个 session 最近的原始消息，供 `mem0` 在抽取新记忆时参考。

## Current Runtime Behavior

### 提示词相关

当前每一轮调用主模型前，项目会：
- 在 prompt 中附带当前中国时间
- 明确要求模型不要把过期记忆当成当前事实
- 把本轮完整提示词打印到控制台

### 记忆检索降级

如果 `mem0.search(...)` 失败：
- 当前轮直接降级为空记忆
- 聊天主流程继续执行
- 不会因为 embedding 检索失败直接把接口打挂

### 记忆保存降级

如果后台批量 `mem0.add(...)` 失败：
- 只打印错误日志
- 不影响主响应

### 滑动窗口与异步策略

当前策略：
- 滑动窗口：同步更新
- `mem0.add(...)`：按批次异步写入线程池

## Public Interfaces

### API

`POST /api/chat` 需要：
- `message`
- `user_id`
- `agent_id`

`GET /api/history/{user_id}` 还需要查询参数：
- `agent_id`

### CLI

启动时显式输入：
- `user_id`
- `agent_id`

## Main Files

### api/routes.py

- `/api/chat`
- `/api/history/{user_id}`

路由层通过 `asyncio.to_thread(...)` 调用同步聊天逻辑，避免阻塞事件循环。

### core/chat_core.py

这是项目最关键的文件，负责：
- 二元作用域会话管理
- 记忆检索
- 续问检索 query 改写
- prompt 组装
- 模型调用
- 提示词打印
- 同步更新滑动窗口
- 5 轮记忆批量缓冲，重叠 2 轮
- 异步保存 `mem0` 长期记忆

### core/mem0_compat.py

负责对 `mem0` 的 `Milvus` 适配器打兼容补丁，关闭不兼容的 `BM25/sparse` 索引逻辑。

### config/settings.py

负责：
- 读取环境变量
- 组装 `Mem0 OSS` 配置
- 组装 `Milvus` 连接参数

## Verification Notes

当前项目已有基础回归测试，主要覆盖：
- 路径与静态目录
- `Mem0 OSS` 配置结构
- `Milvus` dense-only 兼容补丁
- 二元作用域路由参数
- 二元作用域会话隔离
- 二键扁平检索过滤
- 弱语义续问检索改写
- 每 5 轮写入一次，按 2 轮重叠滑动写入
- 检索失败降级
- 保存失败降级
