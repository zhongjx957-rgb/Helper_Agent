# We_Listen 智能客服系统

```
ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
   We_Listen  v2.0
   智能客服 AI 系统
ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
```

## 核心能力

- 三路融合意图识别：LLM 语义理解（权重 70%）+ N-gram 哈希向量匹配（权重 20%，纯 Python 零依赖）+ 关键词模式匹配（权重 10%），加权投票合并，置信度低于阈值降级为 `OTHER`，LLM 与 Embedding 并行调用不串行。
- 多 Agent 路由与编排：三层路由（意图映射 → 同类多实例按在线表现选最优 → 专属 Agent 失败自动降级 GeneralAgent），复杂问题并行派发多个 Agent 合并结果，紧急度/置信度触发升级。
- 三级记忆架构：Redis 工作记忆（毫秒级读写）+ ChromaDB 情景记忆（跨会话语义检索）+ 用户画像（长期偏好提炼），上下文构建时三级融合，工作记忆超阈值自动 LLM 压缩摘要，防止 context 爆炸。
- 混合检索 RAG：BM25 关键词召回（rank_bm25 + jieba 中文分词）与 ChromaDB 向量召回，支持 RRF 与加权两种融合策略；检索链路整合查询改写、并行召回、LLM 重排，召回不全/排序差均有对应优化手段。
- MCP 工具框架：统一工具注册、JSON Schema 参数校验、TTL 结果缓存、三态熔断器、超时重试与降级策略，检索类工具自动走「查询改写 → 并行召回 → 结果重排」优化链路。
- 弹性工具箱：单次调用超时（`call_with_timeout`）、指数退避 + 抖动重试（`with_retry`，只重试瞬态错误）、工具循环守卫（`ToolLoopGuard`，限制模型-工具循环不收敛），链路级超时兜底。
- Skills 热加载：业务话术、处理流程、合规边界、排障 SOP 以 `SKILL.md` 形式随请求动态注入 system prompt，支持运行时热加载，无需重启服务。
- 性能监控与告警：滑动窗口 Z-score 异常检测、阈值告警（日志 + Webhook）、Prometheus 指标导出，Monitor 采集结果写回 Orchestrator 动态调整路由权重，形成「监控 → 反馈 → 路由」闭环。
- 端到端评测：意图识别 Accuracy / Macro-F1、LLM-as-Judge 从相关性/准确性/完整性/有用性四维打分、单轮与多轮对话评测、与历史基线对比的回归检测。
- Anthropic 协议兼容：基于 Anthropic SDK，默认可接入官方 Claude，也可通过 `ANTHROPIC_BASE_URL` 切换兼容协议的三方 API（如 DeepSeek）。

## 技术栈

```text
语言：Python 3.12
Web 框架：FastAPI
服务运行：Uvicorn / ASGI
LLM 接入：Anthropic SDK，兼容 Anthropic 协议三方 API（如 DeepSeek）
短期记忆：Redis
长期记忆：ChromaDB（情景记忆 + 用户画像 + RAG 知识库）
Agent 编排：三层路由多 Agent 编排器
RAG：rank_bm25 + jieba 分词、Chroma 向量、RRF / 加权融合、LLM 重排、查询改写
弹性：超时、指数退避重试、熔断器、TTL 缓存、降级、工具循环守卫
监控：prometheus-client、Z-score 异常检测、Webhook 告警
评测：LLM-as-Judge、Accuracy / F1、回归检测
配置管理：python-dotenv，.env
部署：Docker / Docker Compose（Redis、ChromaDB、Prometheus、Nginx）
```

说明：LLM 调用统一外包 `with_retry`（超时 + 重试），重试耗尽抛 `ResilienceError` 后走各模块既有降级路径；意图识别 LLM 分支失败时由 Embedding / 关键词兜底接管。ChromaDB 优先连接独立服务（docker compose 模式），连不上自动降级为本地嵌入式模式。

## 目录结构

```text
agents/          # 多 Agent 编排器（三层路由、并行协作、降级升级）
api/             # FastAPI 入口与路由
core/            # 意图识别、Skill 加载、弹性工具箱
memory/          # 三级记忆管理（工作 / 情景 / 用户画像）
mcp/             # MCP 工具框架 + RAG 知识库
monitor/         # 性能监控、异常检测、告警
evaluation/      # 端到端评测框架
skills/          # 业务 Skills（支持热加载）
config/          # Prometheus、Nginx 配置
data/            # Chroma 持久化数据、评测基线
tests/           # 单元测试
specs/           # 功能设计文档
```

## Agent loop

每轮对话主链路：记忆读取 → 意图识别 → Agent 路由 → 执行 → 升级检查 → 记忆写入。

```text
收到用户消息
-> 读取三级记忆上下文（Redis 工作记忆 + Chroma 情景记忆 / 用户画像）
-> 意图识别（三路融合加权投票，LLM 与 Embedding 并行）
-> Agent 路由（意图映射 → 性能路由 → 降级 GeneralAgent）
-> 执行（复杂问题并行派发多个 Agent，结果合并）
-> 升级检查（紧急度 / 置信度 / 意图触发转人工）
-> 写入记忆 + 异步更新用户画像
```

各模块分工：

- `IntentRecognizer`：三路融合意图识别，输出意图、置信度、紧急度、实体与理由。
- `GeneralAgent`：通用客服，完整解决、具体操作、结构清晰、覆盖所有子问题。
- `TechnicalAgent`：技术支持，步骤化排障、预期结果说明、兜底方案。
- `BillingAgent`：账单服务，可操作路径、原因解释、时间预期、备选方案。
- `AgentOrchestrator`：维护 Agent 池，负责路由、并行协作、失败降级与升级判定。
- `MemoryManager`：三级记忆读写，工作记忆超阈值时 LLM 压缩并沉淀到情景记忆。

## 安装依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 已包含：

```text
anthropic      # LLM 调用
fastapi        # Web 框架
redis          # 工作记忆
chromadb       # 向量库（情景记忆 + 用户画像 + RAG）
rank_bm25      # BM25 关键词检索
jieba          # 中文分词
prometheus-client
httpx          # Webhook 告警
```

## 环境变量配置

复制 `.env.example` 为 `.env` 并填写实际值：

```bash
cp .env.example .env
```

核心配置：

```env
# LLM（默认官方 Claude，可切换兼容协议三方 API）
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
ANTHROPIC_API_KEY=sk-你的_API_Key
ANTHROPIC_MODEL=deepseek-v4-flash

# Redis（工作记忆）
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=we_listen123

# ChromaDB（情景记忆 + 用户画像 + 知识库）
CHROMA_HOST=localhost
CHROMA_PORT=8001
CHROMA_PERSIST_DIRECTORY=./data/chroma

# Skills
WE_LISTEN_SKILLS_DIR=./skills
WE_LISTEN_SKILLS_MAX_PROMPT_CHARS=5000

# 监控 / 评测
PROMETHEUS_PORT=9091
EVAL_BASELINE_PATH=./data/eval/baseline.json
```

## Docker Compose 一键启动

仓库提供 `Dockerfile` 和 `docker-compose.yml`，会启动：

- `redis`：Redis 7，工作记忆，宿主端口 `6379`
- `chromadb`：ChromaDB 0.5.23，向量库，宿主端口 `8001`
- `prometheus`：Prometheus 指标采集，宿主端口 `9090`
- `we_listen`：We_Listen FastAPI 服务，宿主端口 `8000`
- `nginx`：Nginx 反向代理，宿主端口 `80`

```bash
cp .env.example .env   # 设置 ANTHROPIC_API_KEY 后
docker compose up -d --build
```

服务健康检查：

```bash
curl http://localhost:8000/health
```

也可使用 `docker-deploy.sh` 一键安装/启动/健康检查/备份恢复：

```bash
./docker-deploy.sh install
./docker-deploy.sh start
./docker-deploy.sh health
./docker-deploy.sh backup
```

## 本地开发运行

直接启动 API 服务：

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

交互式 CLI（不依赖 Redis / ChromaDB，适合快速体验）：

```bash
python api/main.py --cli
```

Swagger 文档：`http://localhost:8000/docs`

## 弹性工具箱（超时 / 重试 / 熔断）

LLM 与工具调用统一由 `core/resilience.py` 提供弹性保障：

- `call_with_timeout`：单次调用超时（默认 20s），把「永远等不到」变成「可计数的失败」。
- `with_retry`：指数退避 + 抖动，只重试瞬态错误（超时/网络/限流/5xx），鉴权等永久错误直接抛出；`attempts` 为总尝试次数（默认 3 = 首次 + 2 次重试）。
- `ToolLoopGuard`：限制模型-工具循环总轮次与同参重复调用，防空转烧钱（为后续 function calling 预留）。
- 熔断器（`mcp/tool_manager.py`）：连续失败超阈值进入 `OPEN`，恢复期后 `HALF_OPEN` 探测，防止雪崩。
- 链路级兜底：`AgentOrchestrator.run` 外层 `REQUEST_TOTAL_TIMEOUT`（60s）防止「意图识别 + 多 Agent + 工具调用」叠加超时。

## Skills 热加载

Skills 是可在运行期调整的业务规则，命中关键词时注入对应 Agent 的 system prompt。内置三类：

```text
skills/general_customer_service/SKILL.md  # 通用客服：接待、澄清、分流、投诉和转人工
skills/technical_support/SKILL.md         # 技术支持：故障排查、接口错误、部署配置和安全边界
skills/billing_support/SKILL.md           # 账单服务：扣款、退款、发票、订阅和财务审核
```

修改 Skill 文件后热加载，无需重启：

```bash
curl -X POST http://localhost:8000/skills/reload
curl http://localhost:8000/skills
```

## 混合检索 RAG

知识库基于 ChromaDB（内置 embedding 模型），长文档自动切片（每片 500 字，段落感知 + 句子级递归切分）。检索时 BM25 关键词召回（jieba 分词）与向量召回并行，按策略融合：

```text
RRF 融合：   fusion(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_vector(d))
加权融合：   fusion = alpha * vector_norm + (1 - alpha) * bm25_norm
```

完整检索链路（`/search` 与 `/chat` 共用）：查询改写 → 并行召回 → 合并去重 → LLM 重排 → Top-K。空知识库 / BM25 未就绪时自动回退纯向量或本地检索，不中断服务。

## 调用示例

健康检查：

```bash
curl http://localhost:8000/health
```

对话（主链路：记忆 → 意图识别 → 路由 → 执行 → 记忆写入）：

```bash
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"应用登录一直报 500 错误","user_id":"u_123"}'
```

知识库批量导入文档：

```bash
curl -X POST http://localhost:8000/knowledge/add \
  -H 'Content-Type: application/json' \
  -d '{"documents":[{"title":"退款政策","content":"用户在购买后 7 天内可以申请无理由退款..."}]}'
```

上传文件导入知识库（`.txt` / `.md` / `.json`，≤ 10MB）：

```bash
curl -X POST http://localhost:8000/knowledge/upload \
  -F "file=@docs/refund.md"
```

知识库统计（含 BM25 索引状态）：

```bash
curl http://localhost:8000/knowledge/stats
```

检索优化链路演示（查询改写 → 并行召回 → 重排 → Top-K，可切换融合策略）：

```bash
curl -X POST 'http://localhost:8000/search?query=退款流程&top_k=5&hybrid=true&strategy=rrf'
```

监控摘要与 Prometheus 指标：

```bash
curl http://localhost:8000/monitor
curl http://localhost:8000/metrics
```

## 评测

内置意图识别与单/多轮对话评测用例，开箱即用：

```bash
curl -X POST http://localhost:8000/eval/run
```

支持自定义用例：

```bash
curl -X POST http://localhost:8000/eval/run \
  -H 'Content-Type: application/json' \
  -d '{
    "intent_cases": [{"message":"我的订单什么时候到？","expected_intent":"query"}],
    "dialog_cases": [{"turns":["你好，我想退款","订单号是 #12345","退款多久到账？"]}]
  }'
```

评测结果含通过率、四维平均分、回归项与可操作优化建议；基线写入 `EVAL_BASELINE_PATH`，用于后续回归检测。

## 单元测试

`tests/` 使用 Python 标准库 `unittest`，不依赖 pytest：

```bash
python -m unittest discover -s tests
```

单独运行弹性工具箱用例：

```bash
python -m unittest tests.test_resilience -v
```
