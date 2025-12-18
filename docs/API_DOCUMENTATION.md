# LiveSecBench API 参考

本文档列出在最新代码中最常被复用的 Python API，便于在自定义脚本、CI 流水线或新模块中直接调用核心能力。所有示例默认基于 `python -m pip install -e .` 之后的本地环境。

---

## 目录

1. [配置与任务管理](#1-配置与任务管理)
2. [HTTP 客户端与限流](#2-http-客户端与限流)
3. [批量处理与缓存](#3-批量处理与缓存)
4. [SQLite 存储接口](#4-sqlite-存储接口)
5. [评分框架](#5-评分框架)
6. [日志与辅助工具](#6-日志与辅助工具)
7. [参考资料](#7-参考资料)

---

## 1. 配置与任务管理

### 1.1 ConfigManager
**位置**：`livesecbench/infra/config/config_manager.py`

```python
from livesecbench.infra.config import ConfigManager

cm = ConfigManager("livesecbench/configs/run_custom_safety_benchmark.yaml")

# 基础信息
eval_name = cm.get_eval_run_name()
models = cm.get_models_to_test()            # List[Dict]
dimensions = cm.get_dimensions()            # List[str]
question_selection = cm.get_question_selection()

# 评分 / 存储
elo_settings = cm.get_elo_settings()
scoring_config = cm.get_scoring_config()
storage_type = cm.get_storage_type()        # 'sqlite' 或 'mysql'
db_path = cm.get_storage_db_path()          # SQLite路径
tables = cm.get_storage_tables()            # {'model_outputs_table': 'model_outputs', ...}
mysql_config = cm.get_mysql_config()        # MySQL配置（如使用MySQL）

# API调用设置
api_settings = cm.get_api_call_settings()   # 超时、并发、重试等配置

# 裁判模型配置（自动解析 env_var）
judge_api = cm.get_judge_model_api()

# 模型错误处理配置
error_handlers = cm.get_model_error_handlers()

# 报告设置
report_settings = cm.get_report_settings()
report_template = cm.get_report_prompt_template()

# 模型名称映射
model_name_map = cm.get_model_name_map()    # {model_id: model_name}

# 校验配置
errors = cm.validate_config()
if errors:
    raise ValueError(errors)
```

特点：
- `env_var:VARIABLE_NAME` 会在读取阶段自动解析，不需要手动处理。
- 所有 getter 均返回结构化字段，减少对原始 dict 的依赖。
- 支持 SQLite 和 MySQL 双存储后端。
- 新增 `get_model_name_map()` 方法，便于模型ID到名称的映射。

### 1.2 TaskManager
**位置**：`livesecbench/core/task_manager.py`

```python
from livesecbench.core.task_manager import TaskManager

tm = TaskManager()
tm.set_config_info(config_path="livesecbench/configs/run_custom_safety_benchmark.yaml",
                   eval_run_name="demo_run")
tm.set_models([m["model_name"] for m in models])
tm.set_dimensions(dimensions)
tm.set_question_counts({dim: len(qs) for dim, qs in dimension_questions.items()})

task_info = tm.get_task_info()
print(tm.task_id)  # 例如：20251118_120001
```

---

## 2. HTTP 客户端与限流

**位置**：`livesecbench/infra/http_client.py`

```python
import asyncio
from livesecbench.infra.http_client import RetryableHTTPClient, RateLimiter

async def call_model():
    # 支持每秒请求数、每分钟请求数和每分钟Token数（TPM）限制
    limiter = RateLimiter(
        per_second=5, 
        per_minute=60,
        tokens_per_minute=100000,  # TPM限流
        estimated_tokens_per_request=4000  # 预估每请求token数
    )
    
    client = RetryableHTTPClient(
        base_url="https://api.openai.com/v1",
        api_key="env_var:OPENAI_API_KEY",  # 支持 env_var
        timeout=600,  # 超时时间
        max_retries=5,
        retry_delay=1,
        rate_limiter=limiter,
    )

    resp = await client.post(
        endpoint="chat/completions",
        json_data={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        },
        context_name="demo call",
        task_type="general",  # 任务类型: "general", "judge", "answer"
        identifier={"model": "gpt-4", "dimension": "ethics"},  # 请求标识
    )
    
    # TPM精确统计（在请求完成后调用）
    actual_tokens = resp.get('usage', {}).get('total_tokens', 0)
    await limiter.register_tokens(actual_tokens)
    
    return resp["choices"][0]["message"]["content"]

asyncio.run(call_model())
```

要点：
- 客户端自动记录上下文日志，并在 HTTP 失败时进行指数退避重试。
- `context_name` 会出现在日志中，便于定位具体模型或请求。
- `identifier`参数用于在日志中标识请求来源。
- `task_type`参数区分不同类型的任务。
- 支持TPM（每分钟Token数）限流，采用预估+精确统计策略。
- 自动处理HTTP 204（内容审查）和429（限流）响应。
- **新增**：支持为单个模型设置独立的RPM、TPM和并发限制。
- **新增**：支持模型API切换功能，可在不同API端点间切换。

---

## 3. 批量处理与缓存

### 3.1 BatchProcessor
**位置**：`livesecbench/infra/batch_processor.py`

```python
from livesecbench.infra.batch_processor import BatchProcessor

processor = BatchProcessor(max_concurrent=5, show_progress=True)

async def handle(item):
    ...

result = await processor.process_batch(items=data_list, processor_func=handle)
print(result.success_count, result.failure_count)
```

### 3.2 CacheManager（可选）
**位置**：`livesecbench/infra/cache_manager.py`

```python
from livesecbench.infra.cache_manager import get_cache_manager

cache = get_cache_manager(max_size=1000, enable_disk_cache=False)

@cache.cached(ttl=3600)
def expensive_call(arg):
    ...
```

---

## 4. 存储接口

**位置**：`livesecbench/storage/sqlite_storage.py` 或 `livesecbench/storage/mysql_storage.py`

```python
from livesecbench.storage.sqlite_storage import SQLiteStorage
# 或
from livesecbench.storage.mysql_storage import MySQLStorage

# SQLite 示例
storage = SQLiteStorage(
    db_path="data/livesecbench.db",
    model_outputs_table="model_outputs",
    pk_results_table="pk_results",
    task_id="20251118_120001",
)

# MySQL 示例
storage = MySQLStorage(
    host="localhost",
    user="username",
    password="env_var:MYSQL_PASSWORD",  # 支持 env_var
    database="livesecbench",
    model_outputs_table="model_outputs",
    pk_results_table="pk_results",
    task_id="20251118_120001",
)

# 保存 / 查询模型回答
storage.save_model_output(result_dict)
existing = storage.get_model_output(model="gpt-4", category="ethics", prompt="示例问题")

await storage.asave_model_output(result_dict)
existing = await storage.aget_model_output("gpt-4", "ethics", "示例问题")

# 保存 PK 结果
storage.save_pk_result(
    dimension="ethics",
    question_id="ethics_001",
    model_a="model_a",
    model_b="model_b",
    payload={"winner": "model_a"},
)

# 记录任务摘要
storage.save_task_info(task_id=storage.task_id, task_info=task_manager.get_task_info())
```

常用字段：
- `payload_json`：原始模型响应/PK 详情（JSON 字符串）。
- `consume_time`、`prompt_tokens`、`completion_tokens`：便于统计成本。
- 支持图像信息存储，适配多模态数据集。

**存储工厂**：
```python
from livesecbench.storage import StorageFactory

# 根据配置自动创建适配的存储实例
storage = StorageFactory.create_from_config(config_manager)
```

---

## 5. 评分框架

### 5.1 核心组件
**位置**：`livesecbench/infra/scoring/`

```python
from livesecbench.infra.scoring import (
    SwissPairingStrategy,
    RoundRobinPairingStrategy,
    RandomPairingStrategy,
    ELORatingAlgorithm,
    ConvergenceDetector,
    AdaptiveConvergenceDetector,
    ScoringOrchestrator,
)

pairing = SwissPairingStrategy()
elo = ELORatingAlgorithm(init_rating=1500, k_factor=32, logistic_constant=400)
detector = AdaptiveConvergenceDetector(threshold=0.01, min_rounds=5)

orchestrator = ScoringOrchestrator(
    pairing_strategy=pairing,
    rating_algorithm=elo,
    pk_runner=pk_runner_func,              # 调用裁判模型的协程
    fetch_model_result=fetch_func,         # 读取 SQLite 中的回答
    logger=logger,
    convergence_detector=detector,
)
```

### 5.2 默认评分器（model_based_scorer）
**位置**：`livesecbench/scorers/model_based_scorer.py`

```python
async def score(
    evaluation_dimension: str,
    dimension_questions: list,
    models: list,
    reasoning_models: list,
    scorer_params: dict,
    runtime_context: dict,
) -> dict:
    ...
```

`runtime_context` 提供：
- `logger`
- `pk_runner`：执行裁判模型请求
- `fetch_model_result`：读取模型回答
- `elo_settings`：当前维度的 ELO 配置

自定义评分器可复用上述上下文，只需返回包含 `history_path` / `result_path` / `record_path` 的字典。

### 5.3 排名变化追踪
**位置**：`livesecbench/core/rank.py`

```python
from livesecbench.core.rank import RankManager

rank_manager = RankManager()
# 计算排名变化
rank_changes = rank_manager.calculate_rank_changes(current_results, previous_results)
# 生生排名报告
report = rank_manager.generate_rank_report(rank_changes)
```

---

## 6. 多模态支持

### 6.1 图像输入处理
**位置**：`livesecbench/core/run_model_answer.py`

```python
# 支持多种图像格式：本地文件、URL、base64
image_input = {
    "type": "image_url",
    "image_url": {
        "url": "https://example.com/image.jpg"  # 支持URL格式
    }
}

# 或本地文件
image_input = {
    "type": "image_path",
    "image_path": "/path/to/image.jpg"
}

# 多图像输入
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "描述这张图片"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}}
        ]
    }
]
```

### 6.2 结构化文本输入
**位置**：`livesecbench/core/run_model_answer.py`

```python
# 适配结构化文本格式
structured_input = {
    "question": "原始问题",
    "context": "相关背景信息",
    "metadata": {
        "difficulty": "high",
        "category": "reasoning"
    }
}

# 自动提取子维度问题
sub_dimension_questions = extract_questions_by_sub_dimension(
    dimension="ethics",
    sub_dimension="privacy",
    questions=dataset
)
```

### 6.3 上下文窗口管理
**位置**：`livesecbench/utils/token_util.py`

```python
from livesecbench.utils.token_util import TokenUtil

# 检查并处理上下文溢出
token_util = TokenUtil(model_name="deepseek-chat")
if token_util.is_context_overflow(prompt, max_tokens=32000):
    # 自动截断或重新格式化
    processed_prompt = token_util.truncate_to_fit(prompt, max_tokens=32000)
```

---

## 7. 日志与辅助工具

### 7.1 Logger
**位置**：`livesecbench/utils/logger.py`

```python
from livesecbench.utils.logger import configure_root_logger, get_logger

configure_root_logger(level="INFO", log_to_file=True, log_to_console=True)
logger = get_logger(__name__)

logger.info("message")
logger.error("error", exc_info=True)
```

日志默认输出到控制台及 `livesecbench/logs/YYYY_MM_DD.log`。

### 7.2 环境变量加载
**位置**：`livesecbench/utils/env_loader.py`

`load_project_env()` 会在核心模块 import 时自动执行，支持 `.env` 文件与系统环境变量。

---

## 8. 参考资料

- `README.md` / `README_EN.md`：项目简介与快速开始
- `docs/USER_GUIDE.md`：操作指南与最佳实践
- `docs/ARCHITECTURE.md`：架构与流程详情
- `docs/RESULT_FORMAT.md`：输出文件 & SQLite 字段
- `docs/EXAMPLES.md`：常见脚本与自定义示例
- 论文/技术报告：<https://arxiv.org/abs/2511.02366>

