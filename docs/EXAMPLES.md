# LiveSecBench 使用示例

本文档提供围绕最新代码结构整理的常用示例，便于在脚本、CI 或私有平台中快速复用 LiveSecBench 的核心能力。

---

## 目录

1. [运行完整评测](#1-运行完整评测)
2. [操作配置与环境变量](#2-操作配置与环境变量)
3. [自定义评分与判别逻辑](#3-自定义评分与判别逻辑)
4. [访问结果与数据库](#4-访问结果与数据库)
5. [批量任务与 Mock 演示](#5-批量任务与-mock-演示)
6. [更多参考](#6-更多参考)

---

## 1. 运行完整评测

### 1.1 直接使用 CLI
```bash
python livesecbench/run_livesecbench.py --config livesecbench/configs/run_custom_safety_benchmark.yaml
```

> 确保 `models_to_test` 至少包含两个模型，否则 ELO 对战无法进行。

### 1.2 在自定义脚本中调用
```python
# scripts/run_full_eval.py
import sys
from livesecbench.run_livesecbench import main

if __name__ == "__main__":
    sys.argv = [
        "run_livesecbench",
        "--config",
        "livesecbench/configs/run_custom_safety_benchmark.yaml",
    ]
    main()
```

---

## 2. 操作配置与环境变量

### 2.1 加载并校验配置
```python
from livesecbench.infra.config import ConfigManager

cm = ConfigManager("livesecbench/configs/run_custom_safety_benchmark.yaml")
errors = cm.validate_config()
if errors:
    raise ValueError(errors)

print("Eval:", cm.get_eval_run_name())
print("Models:", len(cm.get_models_to_test()))
print("Dimensions:", cm.get_dimensions())
```

### 2.2 统一管理密钥
```bash
export OPENAI_API_KEY="..."      # 被测模型
export DEEPSEEK_API_KEY="..."    # 裁判模型
export MOCK_ALPHA_API_KEY="..."  # 自定义模型
```
在配置文件中使用 `env_var:OPENAI_API_KEY` 引用，上层代码无需关心明文。

### 2.3 自定义错误处理 / ID 清洗
```yaml
model_error_handlers:
  "gemini-2.0-flash": "ext. PROHIBITED_CONTENT"
  "x-ai/grok-4-fast": "Permission denied"

model_id_prefixes_to_remove:
  - "google/"
  - "anthropic/"
```
`run_model_answer.py` 会自动读取这些字段，在响应缺失或模型 ID 过长时提供 fallback。

---

## 3. 自定义评分与判别逻辑

### 3.1 自定义评分器骨架
```python
# livesecbench/scorers/custom_scorer.py
from typing import Any, Dict, List

async def score(
    evaluation_dimension: str,
    dimension_questions: List[dict],
    models: List[str],
    reasoning_models: List[str],
    scorer_params: dict,
    runtime_context: dict,
) -> Dict[str, Any]:
    logger = runtime_context["logger"]
    pk_runner = runtime_context["pk_runner"]

    logger.info("Custom scoring on %s", evaluation_dimension)
    for question in dimension_questions:
        await pk_runner(
            question=question,
            model_a=models[0],
            model_b=models[1],
            scorer_params=scorer_params,
        )

    return {
        "history_path": "results/custom_history.csv",
        "result_path": "results/custom_rank.csv",
        "record_path": "results/custom_records.xlsx",
    }
```

在配置中启用：
```yaml
scoring_config:
  - dimension: "ethics"
    scorer: "custom_scorer"
    params:
      custom_param: "value"
```

### 3.2 切换裁判模型
```yaml
judge_model_api:
  base_url: "https://api.myjudge.com/v1"
  api_key: "env_var:MY_JUDGE_KEY"
  model: "my-judge-model"
  timeout: 90
  max_retries: 3
  rate_limit_per_second: 3
```

---

## 4. 访问结果与数据库

### 4.1 读取 SQLite
```python
import sqlite3

conn = sqlite3.connect("data/livesecbench.db")
cursor = conn.cursor()

cursor.execute("""
  SELECT model_name, category, status
  FROM model_outputs
  ORDER BY created_at DESC
  LIMIT 10;
""")
print(cursor.fetchall())
conn.close()
```

### 4.2 快速查看任务记录
```bash
sqlite3 data/livesecbench.db \
  "SELECT task_id, config_path, created_at
   FROM evaluation_tasks
   ORDER BY created_at DESC
   LIMIT 5;"
```

### 4.3 解析结果文件
- 综合排名：`results/{date}/{month}-models.csv`
- 统计摘要：`results/{date}/{month}-stats.csv`
- 维度 ELO：`results/{date}/elo_results/{dimension}/`
- 报告：`results/{date}/summary_report*.md`（提示词已嵌入报告，不再输出单独 txt）

详见 `docs/RESULT_FORMAT.md`。

---

## 5. 批量任务与 Mock 演示

### 5.1 批量跑多份配置
```bash
#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "livesecbench/configs/run_custom_safety_benchmark.yaml"
  "configs/privacy_only.yaml"
  "configs/factuality_adversarial.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo "Running $cfg"
  python livesecbench/run_livesecbench.py --config "$cfg"
done
```

### 5.2 Mock 端到端演示
```bash
python scripts/run_mock_e2e.py
```
脚本会：
- 读取 `livesecbench/configs/mock_e2e.yaml`
- Mock 所有 HTTP 请求，返回固定回答/裁判结果
- 在 `mock_results/`、`mock_history/`、`mock_records/` 写入演示文件
- 使用 `data/mock_e2e.db` 存储示例数据

适合在无 API Key 的环境中验证配置或演示报告格式。

### 5.3 pytest 快速验证
```bash
python -m pip install -e .[test]
pytest -k config_manager -v    # 只验证配置解析
pytest -k http_client -v       # 验证 HTTP/重试逻辑
```

---

## 6. 更多参考

- `docs/USER_GUIDE.md`：完整操作流程、最佳实践与故障排查。
- `docs/API_DOCUMENTATION.md`：更细的 API 说明（ConfigManager、HTTP 客户端、评分框架等）。
- `docs/RESULT_FORMAT.md`：输出文件命名、字段与 SQLite 结构。
- `README.md` / `README_EN.md`：项目概览、亮点与快速开始。
- 论文与技术报告：<https://arxiv.org/abs/2511.02366>

