# 结果文件与存储规范

本文档描述 LiveSecBench 在最新版本中的输出目录、命名约定及 SQLite 表结构，便于接入下游 BI、自动化报告或自定义分析脚本。

---

## 目录

1. [目录结构](#1-目录结构)
2. [CSV / Excel 文件说明](#2-csv--excel-文件说明)
3. [SQLite 表结构](#3-sqlite-表结构)
4. [配置中的输出控制](#4-配置中的输出控制)
5. [维度命名与时间戳](#5-维度命名与时间戳)
6. [快速查询示例](#6-快速查询示例)

---

## 1. 目录结构

```
results/
└── {YYYY_MM_DD}/
    ├── {YYYY-MM}-models.csv          # 综合排名（百分制）
    ├── {YYYY-MM}-stats.csv           # 统计摘要
    ├── summary_report.md             # Markdown 报告（含提示词）
    └── elo_results/
        └── {dimension}/
            ├── {dimension}_elo_history_{timestamp}.csv
            ├── {dimension}_elo_raw_{timestamp}.csv
            └── {dimension}_pk_details_{timestamp}.xlsx

data/
└── livesecbench.db                   # SQLite 数据库
    ├── model_outputs
    ├── pk_results
    └── evaluation_tasks
```

说明：
- `elo_results` 目录现与综合结果同属 `results/{date}/`，方便一次性打包分享。
- 报告提示词直接嵌入 `summary_report*.md`，不再生成 `{month}-report-prompt.txt`。

---

## 2. CSV / Excel 文件说明

| 文件 | 位置 | 关键字段 | 用途 |
|------|------|----------|------|
| `{YYYY-MM}-models*.csv` | `results/{date}/` | `rank`, `model_name`, `provider`, `overall_score`, `{dimension}` | 综合排名，百分制得分 |
| `{YYYY-MM}-stats*.csv` | `results/{date}/` | `totalModels`, `averageScore`, `dimensions`, `lastUpdate`, `monthlyIncrease`, `scoreImprovement`, `hasHistory` | 统计摘要，适合仪表盘 |
| `{dimension}_elo_history_{timestamp}.csv` | `results/{date}/elo_results/{dimension}/` | `round` + 每个模型列 | 各轮 ELO 轨迹 |
| `{dimension}_elo_raw_{timestamp}.csv` | 同上 | `rank`, `model`, `elo`, `wins`, `losses`, `win_rate` | 维度最终 ELO 排名 |
| `{dimension}_pk_details_{timestamp}.xlsx` | 同上 | `model_a`, `model_b`, `winner`, `question_text`, `model_a_answer`, `model_b_answer`, `judge_prompt`, `judge_result`, `consume`, `timestamp` 等 | PK 详情（含 reasoning / 参考答案） |

> `summary_report*.md` 包含任务综述、亮点、风险提示及用于生成报告的提示词段落，可直接复制用于再生成。

---

## 3. SQLite 表结构

### 3.1 `model_outputs`
```sql
CREATE TABLE model_outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT,
    model TEXT NOT NULL,
    category TEXT,
    prompt TEXT,
    prompt_hash TEXT,
    status TEXT,
    reasoning TEXT,
    answer TEXT,
    provider TEXT,
    consume_time REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    true_answer TEXT,
    delusion_type TEXT,
    prompt_type TEXT,
    prompt_difficulty TEXT,
    prompt_category TEXT,
    created_at INTEGER,
    current_time TEXT,
    payload_json TEXT NOT NULL,
    UNIQUE(model, category, prompt)
);
```

### 3.2 `pk_results`
```sql
CREATE TABLE pk_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dimension TEXT,
    question_id TEXT,
    model_a TEXT,
    model_b TEXT,
    winner TEXT,
    payload_json TEXT,
    created_at INTEGER,
    UNIQUE(dimension, question_id, model_a, model_b)
);
```

### 3.3 `evaluation_tasks`
记录 `task_id`、配置路径、模型列表、维度、输出目录、完成时间等元数据，便于追溯运行历史。

---

## 4. 配置中的输出控制

```yaml
scoring_settings:
  model_based:
    elo:
      result_output_dir: "elo_results"      # 相对路径会展开为 results/{date}/elo_results/
      history_filename_template: "{dimension}_elo_history_{timestamp}.csv"
      result_filename_template: "{dimension}_elo_raw_{timestamp}.csv"
      record_filename_template: "{dimension}_pk_details_{timestamp}.xlsx"
```

- **相对路径**：以项目根目录为基准，并自动拼接到 `results/{date}/`。
- **绝对路径**：直接写入目标目录，可用于 CI 或远程挂载。

---

## 5. 维度命名与时间戳

| 中文 | 英文（文件/目录） |
|------|-------------------|
| 伦理性 | `ethics` |
| 合法性 | `legality` |
| 事实性 | `factuality` |
| 隐私性 | `privacy` |
| 对抗鲁棒性 | `robustness` |
| 推理安全 | `reasoning` |

- 文件命名统一使用英文小写维度名。
- 时间戳格式：`YYYY_MM_DD_HH_MM_SS`，示例 `2025_11_18_12_00_01`。

---

## 6. 快速查询示例

### 最近 5 次任务
```bash
sqlite3 data/livesecbench.db \
  "SELECT task_id, config_path, created_at
   FROM evaluation_tasks
   ORDER BY created_at DESC
   LIMIT 5;"
```

### 某模型最新回答
```bash
sqlite3 data/livesecbench.db \
  "SELECT model_name, category, status, current_time
   FROM model_outputs
   WHERE model_name='GPT-4'
   ORDER BY created_at DESC
   LIMIT 10;"
```

### 某维度 PK 统计
```bash
sqlite3 data/livesecbench.db \
  "SELECT winner, COUNT(*)
   FROM pk_results
   WHERE dimension='ethics'
   GROUP BY winner
   ORDER BY COUNT(*) DESC;"
```

---

如需扩展新的文件格式或将结果推送到外部存储，可在 `core/rank.py`、`core/report.py` 或 `storage/sqlite_storage.py` 中添加自定义逻辑。更多上下游示例请参考 `docs/EXAMPLES.md`，整体架构见 `docs/ARCHITECTURE.md`。 

