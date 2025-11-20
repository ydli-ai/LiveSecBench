# LiveSecBench 架构设计文档

本文档基于当前代码库（2025‑11）重新梳理 LiveSecBench 的分层架构、核心流程与扩展点，帮助开发者在多次功能迭代后快速对齐系统全貌。

---

## 目录

1. [整体视图](#1-整体视图)
2. [目录与模块职责](#2-目录与模块职责)
3. [核心运行流程](#3-核心运行流程)
4. [数据流与存储](#4-数据流与存储)
5. [扩展能力](#5-扩展能力)
6. [可观测性与恢复](#6-可观测性与恢复)
7. [外部依赖](#7-外部依赖)
8. [参考资料](#8-参考资料)

---

## 1. 整体视图

```
┌────────────────────────────────────────────────────────┐
│  CLI / Scripts                                         │  python livesecbench/run_livesecbench.py
└────────────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────┐
│  Application Layer                                     │
│  • 核心 orchestrator（run_livesecbench.py）            │
│  • 任务管理（core/task_manager.py）                    │
│  • 排名 & 报告（core/rank.py、core/report.py）         │
└────────────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────┐
│  Domain Layer                                           │
│  • 模型回答流水线（core/run_model_answer.py）          │
│  • 评分/裁判器（core/run_scoring.py、scorers/）        │
│  • 题库抽样与维度编排                                  │
└────────────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────┐
│  Infrastructure                                         │
│  • ConfigManager / env loader (infra/config)            │
│  • RetryableHTTPClient & RateLimiter (infra/http)       │
│  • Scoring framework (infra/scoring)                    │
│  • BatchProcessor / CacheManager                        │
│  • SQLiteStorage (storage/sqlite_storage.py)            │
└────────────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────┐
│  External Systems                                       │
│  • 被测 LLM API / 裁判模型 API                          │
│  • 文件系统（results/{date}/…）                         │
│  • SQLite DB：data/livesecbench.db                      │
└────────────────────────────────────────────────────────┘
```

- **模块化**：各层通过显式接口交互，可在不影响上层的情况下替换底层实现。
- **任务可追溯**：任务信息、模型输出与 PK 记录统一持久化，便于断点续跑和审计。
- **自动化产出**：所有榜单、统计与报告统一落在 `results/{date}/`，命名和结构可在配置中覆盖。

---

## 2. 目录与模块职责

```
livesecbench/
├── run_livesecbench.py          # CLI 入口
├── core/
│   ├── run_model_answer.py      # 并发请求被测模型并写入缓存
│   ├── run_scoring.py           # 维度级 orchestrator（配对/裁判/ELO）
│   ├── rank.py                  # 聚合排名 & 统计
│   ├── report.py                # Markdown 报告与提示词
│   └── task_manager.py          # 任务元信息
├── infra/
│   ├── config/                  # ConfigManager、schema 校验与 env 解析
│   ├── scoring/                 # 配对策略、ELO、收敛检测、调度器
│   ├── http_client.py           # RetryableHTTPClient + RateLimiter
│   ├── batch_processor.py       # 协程批处理、失败重试
│   └── cache_manager.py         # 可选缓存（内存/磁盘）
├── storage/
│   └── sqlite_storage.py        # 模型输出、PK、任务记录
├── scorers/
│   └── model_based_scorer.py    # 默认裁判模型评分器
├── question_set/                # 各安全维度题库
├── configs/                     # 运行配置与 mock 配置
├── utils/                       # 日志、环境变量加载
└── scripts/
    └── run_mock_e2e.py          # 离线端到端演示
```

辅助目录：`docs/`（文档）、`tests/`（pytest）、`data/`（SQLite）、`results/`（输出）、`mock_*`（演示输出）。

---

## 3. 核心运行流程

### 3.1 主控流程
1. **初始化**：配置日志，创建 `TaskManager` 并生成 `task_id`。
2. **加载配置/题库**：`ConfigManager` 校验并解析 `question_selection`，按维度过滤 & 抽样题目。
3. **模型回答阶段**：`core/run_model_answer.py` 调度 `RetryableHTTPClient` 并发访问模型 API；命中 SQLite 缓存即跳过。
4. **评分阶段**：`core/run_scoring.py` 针对每个维度启动 orchestrator，执行配对、PK、ELO 更新与可选收敛检测。
5. **聚合输出**：`rank.py` 生成 `{month}-models.csv`、`{month}-stats.csv`；`report.py` 生成 `summary_report*.md`（提示词已嵌入报告，不再另存 `.txt`）。
6. **任务落盘**：输出路径、模型列表、维度题量等信息写入 `evaluation_tasks`，便于追溯。

### 3.2 模型回答流水线
- Payload 支持文本 + 图片（base64 / URL）及可选 `reasoning` 字段。
- `api_call_settings` 控制超时、并发、速率、重试；`RetryableHTTPClient` 自动注入日志上下文。
- 成功与失败的结果都会写入 `model_outputs`，字段包含 reasoning、token 统计与题目信息。

### 3.3 评分/裁判流水线
- 配对策略：瑞士制（默认）、单循环、随机，均位于 `infra/scoring/pairing_strategies.py`。
- 评分算法：`ELORatingAlgorithm`，K 值、初始分、logistic 常数可配置；未来可扩展为 Glicko 等。
- 收敛检测：basic/adaptive 两种实现，基于评分变化率与排名稳定度判断是否提前结束。
- 裁判模型：`scorers/model_based_scorer.py` 将题目与模型回答拼接提示词，调用 `judge_model_api`。
- 输出内容：ELO 历史 CSV、ELO 最终排名 CSV、PK 详情 Excel，均落在 `results/{date}/elo_results/{dimension}/`。

### 3.4 排名与报告
- `rank.py`：归一化多维度得分，生成综合榜单与统计摘要 CSV。
- `report.py`：读取统计数据和维度亮点，输出 Markdown 报告，内嵌生成报告的提示词以便复现。

---

## 4. 数据流与存储

### 4.1 数据流概览
```
配置 / 题库 ──► run_model_answer ──► SQLite(model_outputs)
                               │
                               ├─► run_scoring ──► SQLite(pk_results)
                               │                    │
                               │                    └─► 结果文件 (CSV / XLSX)
                               └─► rank & report ──► results/{date}/
```

### 4.2 文件系统
- `results/{YYYY_MM_DD}/`
  - `{YYYY-MM}-models*.csv`：综合排名（百分制）
  - `{YYYY-MM}-stats*.csv`：统计摘要
  - `summary_report*.md`：Markdown 报告（含提示词）
  - `elo_results/{dimension}/`：ELO 历史/最终排名/PK 详情
- 输出路径、文件模板可通过 `scoring_settings.model_based.elo` 自定义；在 CI 中可定向到 `tmp/` 或挂载目录。

### 4.3 SQLite
- `model_outputs`：模型回答、reasoning、token 统计及题目元信息；`UNIQUE(model, category, prompt)` 实现缓存。
- `pk_results`：维度 + 题目 + 模型组合的 PK 缓存，避免重复裁判。
- `evaluation_tasks`：任务配置、模型列表、维度、输出目录与完成状态。

详细字段及示例请参考 `docs/RESULT_FORMAT.md`。

---

## 5. 扩展能力

| 场景 | 入口 | 说明 |
|------|------|------|
| 自定义题库 / 维度 | `question_set/` + 配置中的 `question_selection` | 直接新增 JSON/CSV 并配置维度名称，可混合显性/隐性场景。 |
| 新评分器 | `scorers/` | 新建 `custom_scorer.py` 并实现 `async score(...)`；配置项 `scorer` 写入文件名即可。 |
| 配对/评分算法 | `infra/scoring/` | 新增策略或算法类并在配置中切换，`ScoringOrchestrator` 自动加载。 |
| 裁判模型切换 | `judge_model_api` | 填写新的 base_url / api_key / model，即可替换为任意 OpenAI-style Chat API。 |
| 结果导出 | `core/report.py`、`core/rank.py` | 可扩展 PDF/HTML 或上传外部存储，任务信息会记录自定义路径。 |
| 调度方式 | `scripts/` 或自定义 CLI | 通过 Bash/Make/CI Pipeline 调度多份配置，也可将核心模块嵌入私有流程。 |

---

## 6. 可观测性与恢复

- **日志**：`utils/logger.py` 同步输出到控制台与 `livesecbench/logs/YYYY_MM_DD.log`，包含 task_id、模型名、维度等上下文。
- **断点续跑**：`SQLiteStorage` 对模型输出与 PK 结果做幂等保存，再次运行同一配置会自动跳过成功记录；如需强制重跑可清理数据库或修改 `eval_run_name`。
- **速率与重试**：`RetryableHTTPClient` 内置 RateLimiter、指数退避重试与错误日志，可精确定位 API 失败原因。
- **健康指标**：收敛检测状态、PK 失败率、模型错误分布等信息写入日志，便于在监控系统中采集。

---

## 7. 外部依赖

| 类型 | 说明 |
|------|------|
| 被测模型 / 裁判模型 | 任意兼容 OpenAI Chat Completions 风格的 HTTP API，凭据通过环境变量注入。 |
| 存储 | 默认仅依赖本地 SQLite；可通过替换 `SQLiteStorage` 或新增导出逻辑接入外部数据库。 |
| 文件系统 | 默认写本地磁盘，可指向挂载盘或对象存储挂载；CI 中常将结果目录设为临时路径。 |

---

## 8. 参考资料

- `README.md` / `README_EN.md`：项目简介与快速开始。
- `docs/USER_GUIDE.md`：详细的运行说明、最佳实践与故障排查。
- `docs/API_DOCUMENTATION.md`：核心 Python API 清单。
- `docs/RESULT_FORMAT.md`：输出文件和 SQLite 字段规范。
- `docs/EXAMPLES.md`：常见脚本、扩展示例与 Mock 流程。
- 研究论文与技术报告：<https://arxiv.org/abs/2511.02366>

