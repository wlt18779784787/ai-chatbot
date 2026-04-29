# 当前项目的存储与检索说明

> 注意：本文档描述的是项目早期 `mem0 OSS + Milvus + SQLite` 的旧架构。当前代码已经切换到 `mem0 平台托管`，不再依赖本地 `Milvus` / `history.db` 作为运行必需项。

这份文档专门解释当前项目里 `mem0`、`Milvus`、`SQLite` 三者分别承担什么角色，为什么你的 `Milvus` 里会看到两个 collection，以及一次聊天请求到底是怎么检索记忆的。

本文基于当前项目代码和当前接入方式说明，不是泛泛而谈的 mem0 官方概念文档。

---

## 1. 项目整体结构

当前项目可以拆成 4 层：

1. Web/API 层
   - 入口在 [api/main.py](/abs/path/C:/Users/sunsh/PycharmProjects/PythonProject1/api/main.py) 和 [api/routes.py](/abs/path/C:/Users/sunsh/PycharmProjects/PythonProject1/api/routes.py)
   - 对外提供网页和 `/api/chat`

2. 聊天编排层
   - 主逻辑在 [core/chat_core.py](/abs/path/C:/Users/sunsh/PycharmProjects/PythonProject1/core/chat_core.py)
   - 负责：
     - 维护滑动窗口
     - 拼提示词
     - 调主聊天模型
     - 异步保存长期记忆

3. 长期记忆层
   - 由 `mem0 OSS` 管理
   - 配置在 [config/settings.py](/abs/path/C:/Users/sunsh/PycharmProjects/PythonProject1/config/settings.py)
   - 向量存储使用 `Milvus`
   - 历史和消息缓存使用 `SQLite`

4. 模型层
  - 主聊天模型：`moonshot/kimi2.5`
   - 记忆 embedding 模型：`qwen/qwen3-embedding-4b`
   - 当前都通过 `OpenRouter` 的 OpenAI-compatible 接口调用

---

## 2. 当前项目里有哪些“记忆”

这个项目里实际上有三种不同层次的“记忆”。

### 2.1 进程内滑动窗口

保存在：
- [core/chat_core.py](/abs/path/C:/Users/sunsh/PycharmProjects/PythonProject1/core/chat_core.py) 的 `UserSession.conversation_window`

特点：
- 只存在 Python 进程内存里
- 服务重启后就没了
- 用来给当前几轮对话补上下文
- 这是“短期记忆”

这部分不进 `Milvus`，也不是你看到的 `SQLite` 主来源。

---

### 2.2 mem0 长期记忆

这部分会被：
- 抽取
- 向量化
- 写入 `Milvus`

这是“长期记忆”。

不是把整段原始对话全文直接存进去，而是 `mem0` 会尝试从一轮对话里抽出值得长期保留的事实。

例如用户说：

```text
我平时只喝无糖可乐，下周要去上海出差。
```

`mem0` 可能会抽成：

- 用户偏好无糖可乐
- 用户下周将去上海出差

然后每一条都作为独立记忆写入向量库。

---

### 2.3 mem0 的 SQLite 辅助数据

这部分不做语义向量检索，但会辅助记忆抽取和记忆历史管理。

当前主要有两张表：

- `history`
- `messages`

后面会详细讲。

---

## 3. 为什么 Milvus 里有两个 collection

你现在看到两个 collection，不是因为项目手写了两个库表，而是 `mem0` 自己有两套向量存储用途。

### 3.1 主 collection

主 collection 名来自配置：
- `MILVUS_COLLECTION_NAME`

你当前项目默认是：

```text
mem0_qwen4b
```

这个 collection 的作用是：
- 存“正式长期记忆”
- 每一条记录对应一条抽取后的 memory

这才是聊天时最主要使用的那个向量库集合。

---

### 3.2 第二个 collection：`_entities` 后缀

第二个 collection 不是你项目自己命名的，而是 `mem0` 的 entity store 自动派生出来的。

在 `mem0` 的 `Memory.entity_store` 逻辑里，会这么做：

- 复制一份当前 vector store 配置
- 把 `collection_name` 改成：

```text
{主 collection 名}_entities
```

也就是如果主 collection 是：

```text
mem0_qwen4b
```

那么第二个 collection 就是：

```text
mem0_qwen4b_entities
```

这段逻辑在 `mem0` 包的 `memory/main.py` 里。

它的用途不是存正式记忆文本，而是存“实体索引”。

---

### 3.3 entity collection 里存什么

entity collection 里每条记录大致是这种结构：

- `data`
  - 实体文本本身
  - 例如人名、地名、组织名、某个关键词
- `entity_type`
  - 实体类型
- `linked_memory_ids`
  - 这个实体关联到哪些 memory_id
- `user_id` / `agent_id` / `run_id`
  - 作用域信息

它的目标是：
- 把“记忆内容”和“里面出现的实体”再建一层索引关系
- 检索时如果 query 里抽出了类似实体，就能对相关记忆做额外加权

简单说：

- 主 collection：存“记忆本体”
- `_entities` collection：存“记忆里抽出来的实体索引”

---

## 4. 当前项目中两个 collection 的关系

你可以把这两个 collection 理解成下面这种关系：

### 4.1 主 collection：长期记忆正文

例如某条记忆：

```text
用户喜欢无糖可乐
```

这条会写到主 collection，带上自己的 embedding 和 metadata。

---

### 4.2 entity collection：辅助召回索引

如果 `mem0` 从这条记忆里识别出实体或关键词，例如：

- 无糖可乐

那么 entity collection 里可能会有一条实体记录，记录：

- `data = "无糖可乐"`
- `linked_memory_ids = [这条记忆的 memory_id]`

以后用户再问：

```text
你记得我平时喝什么吗
```

或者问到：

```text
可乐
```

这层实体索引可能帮助 `mem0` 对相关记忆做加权。

---

## 5. 你当前项目对 Milvus 做过什么特殊处理

原始 `mem0` 的 `Milvus` 适配器会默认创建混合检索结构：

- dense 向量字段 `vectors`
- 文本字段 `text`
- sparse 字段 `sparse`
- BM25 function
- sparse inverted index

但你的远程 `Milvus` 不支持这套 BM25 sparse 索引参数，所以项目在：

- [core/mem0_compat.py](/abs/path/C:/Users/sunsh/PycharmProjects/PythonProject1/core/mem0_compat.py)

里做了兼容补丁。

补丁后的结果是：

- 只保留 dense 向量索引
- 不再创建 `text`
- 不再创建 `sparse`
- 不再创建 BM25 索引

所以你现在的 `Milvus` 使用方式是：

- 主体仍然是 `mem0`
- 但检索模式已经退化为“纯 dense 向量检索”

这点非常重要，因为它会影响你对“为什么有两个 collection”的理解：

- 现在两个 collection 仍然存在
- 但它们都是 dense vector store 逻辑
- 已经不是 “main dense + sparse BM25” 的结构

---

## 6. 主 collection 里到底写入了什么

当前项目调用 `mem0.add(...)` 时，传入的是：

```python
[
    {"role": "user", "content": user_input},
    {"role": "assistant", "content": assistant_output},
]
```

也就是一整轮问答。

`mem0` 内部处理大致是：

1. 取当前 session 最近消息
2. 取当前用户已有相关记忆
3. 用内部 LLM 做记忆抽取
4. 对抽取出的每条记忆文本做 embedding
5. 给每条记忆生成 metadata
6. 批量写入向量库

主 collection 每条记录核心会包含：

- `id`
  - 记忆的 UUID
- `vectors`
  - 这条记忆对应的 embedding 向量
- `metadata`
  - 业务字段都在这里

`metadata` 当前通常包含：

- `data`
  - 记忆文本正文
- `text_lemmatized`
  - 预处理后的文本
- `hash`
  - 文本 hash，用来去重
- `created_at`
  - 创建时间
- `updated_at`
  - 更新时间
- `user_id`
  - 属于哪个用户
- 可能还有：
  - `agent_id`
  - `run_id`
  - `attributed_to`
  - 其他 metadata

对当前项目来说，最关键的是：

- `data`
- `user_id`
- `created_at`
- `updated_at`

因为聊天时真正取回来塞进提示词里的，主要就是 `data`。

---

## 7. SQLite 里有什么

当前 `SQLite` 文件路径来自：
- `MEM0_HISTORY_DB_PATH`

你当前项目默认是：

```text
history.db
```

里面至少有两张表。

---

### 7.1 history 表

作用：
- 记录“某条记忆对象经历了什么变化”

表结构核心字段：

- `id`
- `memory_id`
- `old_memory`
- `new_memory`
- `event`
- `created_at`
- `updated_at`
- `is_deleted`
- `actor_id`
- `role`

你可以把它看成“记忆审计日志”。

例如：
- 新增一条记忆时，通常写一条 `ADD`
- 更新记忆时，记录旧值和新值
- 删除记忆时，也会留下记录

当前你的项目最常发生的是：
- `ADD`

因为你当前主路径主要在做新增和检索，还没专门做记忆编辑后台。

---

### 7.2 messages 表

作用：
- 存最近原始消息缓存
- 不是向量检索库
- 主要给 `mem0` 记忆抽取提供上下文

字段包括：

- `id`
- `session_scope`
- `role`
- `content`
- `name`
- `created_at`

`mem0` 在执行 `add(...)` 时，会先读取这个 session 最近 10 条消息，再结合当前新增消息，做本轮记忆抽取。

也就是说，这张表不是给聊天前端直接展示用的，而是给 `mem0` 的“抽取流水线”用的。

---

## 8. 当前项目里，SQLite 和 Milvus 分别负责什么

一句话分工：

### 8.1 Milvus 负责“语义记忆检索”

它解决的问题是：

- 这个用户以前说过什么
- 当前问题和哪些长期记忆最相关

它返回的是：

- 若干条候选记忆
- 每条带相似度
- 然后项目主要取记忆文本塞进 prompt

---

### 8.2 SQLite 负责“辅助上下文”和“记忆变更历史”

它解决的问题是：

- 最近几条原始消息是什么
- 这条记忆是什么时候加的、改过没有、删过没有

所以 SQLite 不是本项目里“用来回答用户问题的主数据库”，它更像辅助数据库。

---

## 9. 两个数据库在检索时是怎么配合的

这是最容易混淆的部分。

### 9.1 聊天前的记忆检索

当用户发来一条消息，当前项目会先执行：

- `chat_core.search_memories(query, user_id)`

这一步本质上走的是：

- `mem0.search(...)`

在 `mem0` 内部，大致会这样做：

1. 先把 query 做 embedding
2. 去主 collection 做向量搜索
3. 如有启用 entity store，还可能额外做 entity boost
4. 合并结果后返回最相关记忆

所以聊天前这一步，主检索数据库其实是：

- `Milvus 主 collection`

SQLite 不直接参与“向量召回主结果”。

---

### 9.2 entity collection 在检索中的作用

如果 `mem0` 在 query 里抽到了实体，它可能会：

1. 对实体文本做 embedding
2. 去 `_entities` collection 搜
3. 找到关联的 `linked_memory_ids`
4. 对这些 memory_id 对应的正式记忆做额外加权

所以 `_entities` collection 不是主召回集合，而是“召回增强集合”。

它更像一个：

- 语义实体倒排索引

---

### 9.3 SQLite 在写入时的作用比在检索时更大

SQLite `messages` 表更重要的时机不是聊天前，而是：

- `mem0.add(...)` 时

当你异步保存一轮新记忆时，`mem0` 会：

1. 先从 SQLite `messages` 拿最近 10 条消息
2. 再看这轮新消息
3. 再查 Milvus 里已有相关记忆
4. 再让 LLM 决定要不要形成新的长期记忆

所以写入长期记忆时，实际上是：

- SQLite `messages`
- Milvus 主 collection
- 可能还有 entity collection

一起参与。

---

## 10. 当前项目一次完整聊天时的数据库交互顺序

下面按时间顺序讲一遍。

### 第一步：用户发消息

前端调用：

- `POST /api/chat`

---

### 第二步：先查长期记忆

项目会调用：

- `mem0.search(...)`

这一步主要发生的是：

- 用 embedding 模型把 query 转向量
- 去 `Milvus 主 collection` 搜相似记忆
- 如需要，利用 `_entities` collection 做实体加权

返回的结果会被压缩成若干条 `memory` 文本，注入到 prompt 里。

---

### 第三步：主模型回答

主聊天模型看到的信息包括：

- 当前时间
- 角色人设
- 从 `Milvus` 召回的长期记忆
- 本进程内的滑动窗口
- 用户当前输入

它先产出回复。

---

### 第四步：同步更新滑动窗口

这是项目自己维护的，不是 `mem0`。

---

### 第五步：异步保存长期记忆

后台线程里调用：

- `mem0.add(...)`

这一步发生的事最复杂：

1. 从 SQLite `messages` 拿最近 10 条原始消息
2. 把这一轮新问答也纳入分析
3. 去 `Milvus 主 collection` 查已有相关记忆
4. 用 LLM 抽取应该保存的长期事实
5. 对每条事实做 embedding
6. 写入 `Milvus 主 collection`
7. 抽实体并写入 `Milvus _entities collection`
8. 在 SQLite `history` 写记忆历史
9. 在 SQLite `messages` 写最近消息缓存

所以你看到的两个 collection 和 SQLite，实际上都在“异步写记忆”这一步发生作用。

---

## 11. 当前项目到底可以取出什么

### 11.1 从主 collection 可以取出

- 某个用户的长期记忆
- 每条记忆的正文
- 相似度分数
- 记忆创建/更新时间
- 作用域 metadata

当前项目真正用的是：

- 记忆正文 `memory`

---

### 11.2 从 entity collection 可以取出

- 某个实体记录
- 它关联了哪些 memory_id
- 它属于哪个 user_id / run_id / agent_id

当前项目没有直接把这部分展示给前端，但 `mem0` 内部可能用它来增强召回。

---

### 11.3 从 SQLite history 可以取出

- 某条记忆经历过哪些变更
- 初始内容是什么
- 更新后变成什么
- 是什么时候发生的

如果以后你做“记忆管理后台”，这张表很有用。

---

### 11.4 从 SQLite messages 可以取出

- 某个 session_scope 最近若干条原始消息

这张表主要给 `mem0` 内部用，不是你当前前端历史展示的主要来源。

---

## 12. 当前项目要注意的现实限制

### 12.1 滑动窗口和 SQLite messages 不是同一回事

很多人会误以为：

- 前端历史 = SQLite `messages`

但当前项目不是这样。

前端历史接口：
- 读的是 `UserSession.conversation_window`

SQLite `messages`：
- 是 `mem0` 内部最近消息缓存

两者用途不同。

---

### 12.2 主 collection 和 entity collection 都是 mem0 自动维护的

你项目没有手写“插入第二个 collection”的逻辑。

只要 `mem0` 走到 entity store 路径，它就可能自动创建：

```text
{主 collection 名}_entities
```

---

### 12.3 当前已经关闭了 BM25 稀疏索引

所以你现在不要把第二个 collection 误解成：

- 一个 dense
- 一个 sparse

不是。

当前更准确的理解是：

- 一个是记忆本体向量库
- 一个是实体索引向量库

---

## 13. 用最简短的话总结

如果只记一句话，就记这个：

当前项目里：

- `Milvus 主 collection` 存的是“抽取后的长期记忆正文 + 向量 + metadata”
- `Milvus _entities collection` 存的是“从记忆里抽出的实体 + 它关联了哪些 memory_id”
- `SQLite history` 存的是“记忆对象的增删改历史”
- `SQLite messages` 存的是“最近原始消息缓存，供 mem0 抽取新记忆时参考”

聊天时主要先查的是：

- `Milvus 主 collection`

写长期记忆时会综合使用：

- `Milvus 主 collection`
- `Milvus _entities collection`
- `SQLite messages`
- `SQLite history`

---

## 14. 结合你当前项目的具体名字

按你当前配置，最可能看到的是：

- 主 collection：

```text
mem0_qwen4b
```

- 实体 collection：

```text
mem0_qwen4b_entities
```

- SQLite 文件：

```text
history.db
```

如果你后面改了：

- `MILVUS_COLLECTION_NAME`
- `MEM0_HISTORY_DB_PATH`

那这些名字会跟着变化，但职责不变。
